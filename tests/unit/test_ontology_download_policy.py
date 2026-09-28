"""Where this server may fetch an ontology from, and where the rule is enforced.

`PLAN_ONTOLOGY_PHASE2.md` §3.5 and §3.5.1; acceptance 21-29.

**The rule has to live at the request, not at the setting**, and the reason is
specific to this project rather than general caution: every source configured
here is an OBO Foundry PURL, and a PURL *is* a redirect. A check performed when
the list is read has already passed by the time the 302 arrives. Measured on
this interpreter, `HTTPRedirectHandler.http_error_302` permits targets whose
scheme is in ``('http', 'https', 'ftp', '')`` — so `https` → `ftp` is followed.

**No test here touches DNS or any host.** A stub resolver supplies the
addresses, which is what makes "a host resolving to several addresses, one of
them private" a case that can be written at all.

Module: tests/unit/test_ontology_download_policy.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ontology.download import (  # noqa: E402
    ALLOWED_SCHEMES,
    DestinationPolicy,
    OntologyDownloadError,
    check_destination,
    download_ontology,
)
from src.ontology.settings import (  # noqa: E402
    DEFAULT_ONTOLOGY_SOURCES,
    OntologySettingsError,
    load_ontology_settings,
)

PUBLIC = "93.184.216.34"
PRIVATE = "10.1.2.3"
LOOPBACK = "127.0.0.1"
METADATA = "169.254.169.254"


def policy(mapping=None, *, allowed=()):
    """A policy whose DNS is a dictionary."""
    mapping = mapping or {}
    return DestinationPolicy(
        allowed_hosts=tuple(allowed),
        resolver=lambda host: mapping.get(host, [PUBLIC]),
    )


class TestTheSchemeRule:

    def test_a_permitted_scheme_passes(self):
        check_destination("https://example.org/mondo.obo", policy())

    @pytest.mark.parametrize("url", [
        "ftp://example.org/mondo.obo",
        "file:///etc/passwd",
        "data:text/plain,hello",
        "example.org/mondo.obo",
    ])
    def test_anything_else_is_refused(self, url):
        """`urlretrieve`'s opener carries `FileHandler`, `FTPHandler` and
        `DataHandler`, so a field named "download URL" accepts more than a URL
        unless it says otherwise."""
        with pytest.raises(OntologyDownloadError, match="scheme"):
            check_destination(url, policy())

    def test_a_url_with_no_host_is_refused(self):
        with pytest.raises(OntologyDownloadError, match="no host"):
            check_destination("https:///mondo.obo", policy())

    def test_the_settings_reader_asks_the_same_module(self, tmp_path):
        """**Acceptance 21**, and one rule rather than two: the config-time
        check imports `ALLOWED_SCHEMES` instead of restating it."""
        config = tmp_path / "deployment.yaml"
        config.write_text(
            "ontology:\n  sources:\n    mondo:\n      - ftp://example.org/mondo.obo\n"
        )

        with pytest.raises(OntologySettingsError) as caught:
            load_ontology_settings(config)

        assert "ftp://example.org/mondo.obo" in str(caught.value)
        assert "ontology.sources.mondo" in str(caught.value)


class TestTheDestinationRule:

    def test_a_public_address_is_permitted(self):
        """**Acceptance 27's mirror**: without this the default would be a
        blanket block and every refusal below would pass vacuously."""
        check_destination("https://purl.obolibrary.org/obo/mondo.obo",
                          policy({"purl.obolibrary.org": [PUBLIC]}))

    @pytest.mark.parametrize("address,what", [
        (PRIVATE, "private"),
        (LOOPBACK, "loopback"),
        (METADATA, "link-local"),
        ("::1", "IPv6 loopback"),
        ("fd00::1", "IPv6 private"),
        ("0.0.0.0", "unspecified"),
    ])
    def test_an_address_inside_the_network_is_refused(self, address, what):
        """**Acceptance 25.** `is_global` is one predicate over both families."""
        with pytest.raises(OntologyDownloadError, match="globally routable"):
            check_destination("https://mirror.internal/mondo.obo",
                              policy({"mirror.internal": [address]}))

    def test_a_literal_private_address_is_refused(self):
        """**Acceptance 26.** It has no hostname to match against a list, which
        is why the rule is on addresses rather than on spelling."""
        with pytest.raises(OntologyDownloadError, match="globally routable"):
            check_destination(f"http://{METADATA}/latest/meta-data/", policy())

    def test_an_allowed_host_is_permitted_inside_the_network(self):
        """**Acceptance 27.** An in-house mirror is legitimate and probably
        desirable; the entry is what makes it deliberate."""
        check_destination(
            "https://ontology-mirror.hospital.internal/mondo.obo",
            policy({"ontology-mirror.hospital.internal": [PRIVATE]},
                   allowed=("ontology-mirror.hospital.internal",)),
        )

    def test_the_allow_list_is_not_case_sensitive(self):
        check_destination(
            "https://Mirror.Internal/mondo.obo",
            policy({"Mirror.Internal": [PRIVATE]}, allowed=("mirror.internal",)),
        )

    def test_one_private_answer_among_several_refuses(self):
        """**Acceptance 29, and the whole gap in one line.** A name resolving to
        a public address *and* a private one is a name that can reach the
        private one; admitting it because the first answer was fine is exactly
        the mistake."""
        with pytest.raises(OntologyDownloadError, match="globally routable"):
            check_destination("https://split.example/mondo.obo",
                              policy({"split.example": [PUBLIC, PRIVATE]}))

    def test_a_host_that_resolves_to_nothing_is_refused(self):
        """Unknown is not permitted: where the request would go is unknown,
        which is not the same as its being acceptable."""
        with pytest.raises(OntologyDownloadError, match="no address"):
            check_destination("https://nowhere.example/x.obo",
                              policy({"nowhere.example": []}))

    def test_an_unclassifiable_answer_is_refused(self):
        with pytest.raises(OntologyDownloadError, match="classify"):
            check_destination("https://odd.example/x.obo",
                              policy({"odd.example": ["not-an-address"]}))


class TestTheRedirectIsWhereTheRuleActuallyBites:
    """Acceptance 22, 23 and 28. The stdlib handler is the measured hazard."""

    @staticmethod
    def _handler(pol):
        from src.ontology.download import _GuardedRedirectHandler

        return _GuardedRedirectHandler(pol)

    class _Resp:
        headers: dict = {}

        def geturl(self):
            return "https://purl.example/obo/mondo.obo"

    def test_the_stdlib_would_follow_https_to_ftp(self):
        """The premise, measured here so the guard below is not defending
        against something that never happens."""
        import inspect
        import re

        source = inspect.getsource(urllib.request.HTTPRedirectHandler.http_error_302)
        permitted = re.search(r"urlparts\.scheme not in \(([^)]*)\)", source)

        assert permitted is not None
        assert "'ftp'" in permitted.group(1), (
            "this interpreter no longer permits ftp redirect targets; the guard "
            "is still correct but this test's premise has changed"
        )

    def test_a_redirect_to_ftp_is_refused(self):
        """**Acceptance 22.** The config-time check passed long ago."""
        req = urllib.request.Request("https://purl.example/obo/mondo.obo")

        with pytest.raises(OntologyDownloadError, match="scheme"):
            self._handler(policy()).redirect_request(
                req, self._Resp(), 302, "Found", {}, "ftp://elsewhere.example/mondo.obo"
            )

    def test_a_redirect_into_the_network_is_refused(self):
        """**Acceptance 28**, the destination twin of the case above."""
        req = urllib.request.Request("https://purl.example/obo/mondo.obo")
        pol = policy({"purl.example": [PUBLIC], "inside.example": [PRIVATE]})

        with pytest.raises(OntologyDownloadError, match="globally routable"):
            self._handler(pol).redirect_request(
                req, self._Resp(), 302, "Found", {}, "https://inside.example/mondo.obo"
            )

    def test_an_ordinary_redirect_is_still_followed(self):
        """**Acceptance 23.** Every source this project ships is a PURL, so a
        guard that blocked redirects would block everything."""
        req = urllib.request.Request("https://purl.example/obo/mondo.obo")
        pol = policy({"purl.example": [PUBLIC], "github.example": [PUBLIC]})

        result = self._handler(pol).redirect_request(
            req, self._Resp(), 302, "Found", {}, "https://github.example/mondo.obo"
        )

        assert result is not None
        assert result.full_url == "https://github.example/mondo.obo"

    def test_the_chain_is_bounded(self):
        from src.ontology.download import MAX_REDIRECTS

        handler = self._handler(policy())
        req = urllib.request.Request("https://purl.example/obo/mondo.obo")

        with pytest.raises(OntologyDownloadError, match="redirects"):
            for hop in range(MAX_REDIRECTS + 2):
                handler.redirect_request(
                    req, self._Resp(), 302, "Found", {},
                    f"https://hop{hop}.example/mondo.obo",
                )


class TestTheOneFetchPath:

    def test_a_refused_source_writes_nothing(self, tmp_path):
        target = tmp_path / "mondo.obo"

        with pytest.raises(OntologyDownloadError):
            download_ontology("ftp://example.org/mondo.obo", target, policy=policy())

        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_a_refusal_leaves_an_existing_file_untouched(self, tmp_path):
        """A refused fetch is not a reason to truncate what is already serving."""
        target = tmp_path / "mondo.obo"
        target.write_bytes(b"the file that is already here")

        with pytest.raises(OntologyDownloadError):
            download_ontology("https://blocked.example/mondo.obo", target,
                              policy=policy({"blocked.example": [PRIVATE]}))

        assert target.read_bytes() == b"the file that is already here"

    def test_a_permitted_fetch_writes_the_bytes(self, tmp_path):
        """With a stub opener, so nothing leaves this process."""
        target = tmp_path / "mondo.obo"
        payload = b"format-version: 1.2\n"

        class Stub:
            def open(self, url, timeout=None):
                class Body:
                    def __init__(self):
                        self._data = payload

                    def read(self, n):
                        out, self._data = self._data[:n], self._data[n:]
                        return out

                    def __enter__(self):
                        return self

                    def __exit__(self, *a):
                        return False

                return Body()

        written = download_ontology(
            "https://purl.example/obo/mondo.obo", target,
            policy=policy({"purl.example": [PUBLIC]}),
            opener_factory=lambda *handlers: Stub(),
        )

        assert written.read_bytes() == payload

    def test_a_failed_transfer_leaves_no_partial_file(self, tmp_path):
        target = tmp_path / "mondo.obo"

        class Stub:
            def open(self, url, timeout=None):
                raise OSError("connection reset")

        with pytest.raises(OntologyDownloadError, match="failed"):
            download_ontology("https://purl.example/obo/mondo.obo", target,
                              policy=policy({"purl.example": [PUBLIC]}),
                              opener_factory=lambda *h: Stub())

        assert not target.exists()
        assert not [p for p in tmp_path.iterdir() if p.name.endswith(".part")]

    def test_the_owl_fallback_has_no_fetch_of_its_own(self):
        """**Acceptance 24.** A fallback with its own fetch is the gate not
        existing — and the fallback is the path a rotted PURL leads to, so it
        is the one that most needs the rules."""
        import inspect

        from src.ontology import loader

        source = inspect.getsource(loader)
        assert "urlretrieve" not in source, (
            "the loader fetches directly again, bypassing the scheme and "
            "destination rules"
        )
        body = inspect.getsource(loader.OntologyLoader._download_ontology)
        calls = [line for line in body.splitlines() if "download_ontology(url" in line]
        assert len(calls) == 1, f"more than one fetch call in the download path: {calls}"


class TestTheSettingsContract:

    def test_an_absent_file_is_absence_not_an_error(self, tmp_path):
        settings = load_ontology_settings(tmp_path / "nothing.yaml")

        assert settings.roots == ()
        assert settings.urls_for("mondo") == DEFAULT_ONTOLOGY_SOURCES["mondo"]

    def test_relative_roots_resolve_against_the_repository(self, tmp_path):
        """A build run from a subdirectory has to find the same files as one
        run from the root, so the base is stated rather than inherited from
        the working directory."""
        from src.ontology.settings import REPO_ROOT

        config = tmp_path / "deployment.yaml"
        config.write_text("paths:\n  ontology_roots:\n    - data/ontologies/\n")

        settings = load_ontology_settings(config)

        assert settings.roots == (REPO_ROOT / "data/ontologies",)

    def test_an_absolute_root_is_left_alone(self, tmp_path):
        config = tmp_path / "deployment.yaml"
        config.write_text(f"paths:\n  ontology_roots:\n    - {tmp_path / 'onts'}\n")

        assert load_ontology_settings(config).roots == (tmp_path / "onts",)

    def test_a_present_but_malformed_key_raises(self, tmp_path):
        """**Absent and wrong are different states** and only one of them is a
        deployment as intended."""
        config = tmp_path / "deployment.yaml"
        config.write_text("paths:\n  ontology_roots: data/ontologies/\n")

        with pytest.raises(OntologySettingsError, match="must be a list"):
            load_ontology_settings(config)

    def test_a_configured_source_replaces_the_default(self, tmp_path):
        """**Acceptance 6's config half**: editing a URL here changes what a
        download attempts, with no code change."""
        config = tmp_path / "deployment.yaml"
        config.write_text(
            "ontology:\n  sources:\n    mondo:\n"
            "      - https://mirror.example/mondo.obo\n"
        )

        settings = load_ontology_settings(config)

        assert settings.urls_for("mondo") == ("https://mirror.example/mondo.obo",)
        assert settings.urls_for("hpo") == DEFAULT_ONTOLOGY_SOURCES["hpo"], (
            "listing one ontology should not silently drop the others"
        )

    def test_allowed_hosts_are_read(self, tmp_path):
        config = tmp_path / "deployment.yaml"
        config.write_text("ontology:\n  allowed_hosts:\n    - mirror.internal\n")

        assert load_ontology_settings(config).allowed_hosts == ("mirror.internal",)

    def test_the_committed_configuration_still_parses(self):
        """The file this project actually ships, read through this reader."""
        settings = load_ontology_settings()

        assert settings.urls_for("mondo")
        for name in DEFAULT_ONTOLOGY_SOURCES:
            for url in settings.urls_for(name):
                assert url.split(":", 1)[0] in ALLOWED_SCHEMES


class TestATruncatedTransferIsNotAFinishedOne:
    """The check `urlretrieve` had and this downloader dropped.

    `HTTPResponse.read(amt)` returns `b""` when the connection closes early —
    it does not raise — so a short read ends the loop exactly like a complete
    one, and temp-and-replace then publishes the fragment over a good file. The
    cut can land on a stanza boundary, in which case the fragment parses, passes
    the role check, and has its digest recorded as the input.
    """

    @staticmethod
    def _opener(body: bytes, declared: int | None):
        """A real `http.client.HTTPResponse` over a controlled socket.

        Not a stub that raises: the whole point is that nothing raises. A test
        whose opener threw `OSError` would pass against the broken version.
        """
        import http.client
        import io

        header = b"HTTP/1.1 200 OK\r\n"
        header += (
            f"Content-Length: {declared}\r\n".encode()
            if declared is not None
            else b"Transfer-Encoding: chunked\r\n"
        )
        raw = header + b"\r\n" + body

        class Sock:
            def __init__(self):
                self._f = io.BytesIO(raw)

            def makefile(self, *a, **k):
                return self._f

        class Opener:
            def open(self, url, timeout=None):
                response = http.client.HTTPResponse(Sock(), method="GET")
                response.begin()
                return response

        return lambda *handlers: Opener()

    def test_a_short_body_is_refused(self, tmp_path):
        target = tmp_path / "mondo.obo"

        with pytest.raises(OntologyDownloadError, match="delivered"):
            download_ontology(
                "https://purl.example/obo/mondo.obo", target,
                policy=policy({"purl.example": [PUBLIC]}),
                opener_factory=self._opener(b"format-version: 1.2\n", 120),
            )

        assert not target.exists()

    def test_the_file_already_in_place_survives_it(self, tmp_path):
        """**The damage the missing check actually did.** A stanza-aligned cut
        parses, so the fragment would have replaced a complete ontology and
        been recorded as the input."""
        target = tmp_path / "mondo.obo"
        good = b"format-version: 1.2\nontology: mondo\n\n[Term]\nid: MONDO:1\nname: t\n"
        target.write_bytes(good)

        with pytest.raises(OntologyDownloadError):
            download_ontology(
                "https://purl.example/obo/mondo.obo", target,
                policy=policy({"purl.example": [PUBLIC]}),
                opener_factory=self._opener(b"format-version: 1.2\n[Term]\n", 900),
            )

        assert target.read_bytes() == good
        assert not [p for p in tmp_path.iterdir() if p.name.endswith(".part")]

    def test_a_complete_body_is_accepted(self, tmp_path):
        """Or the check above would be satisfied by refusing everything."""
        target = tmp_path / "mondo.obo"
        body = b"format-version: 1.2\n"

        download_ontology(
            "https://purl.example/obo/mondo.obo", target,
            policy=policy({"purl.example": [PUBLIC]}),
            opener_factory=self._opener(body, len(body)),
        )

        assert target.read_bytes() == body

    def test_a_response_declaring_no_length_is_not_failed_for_it(self, tmp_path):
        """**Not possible to check is not the same as checked.** A chunked
        response carries no `Content-Length`, and treating the absence as zero
        would make every such transfer pass a check that never ran — while
        treating it as failure would refuse a legitimate server."""
        target = tmp_path / "mondo.obo"

        download_ontology(
            "https://purl.example/obo/mondo.obo", target,
            policy=policy({"purl.example": [PUBLIC]}),
            opener_factory=self._opener(b"0\r\n\r\n", None),
        )

        assert target.exists()


import http.client as _http_client  # noqa: E402
import io as _io  # noqa: E402
import socket as _socket  # noqa: E402


def _http(raw: bytes, url: str):
    """A real `http.client.HTTPResponse` over bytes, shaped as urllib hands it on."""
    class Sock:
        def __init__(self):
            self._f = _io.BytesIO(raw)

        def makefile(self, *a, **k):
            return self._f

    response = _http_client.HTTPResponse(Sock(), method="GET")
    response.begin()
    response.url = url
    response.msg = response.reason
    return response


class _CannedTransport(urllib.request.BaseHandler):
    """Serves canned responses by URL, ahead of the real HTTP handlers.

    Everything else in the opener is real — including the guarded redirect
    handler `download_ontology` installs — which is the point: the guard was
    only ever tested on its own, never shown to be in the path.
    """

    handler_order = 100

    def __init__(self, responses):
        self.responses = responses
        self.opened = []

    def _serve(self, req):
        self.opened.append(req.full_url)
        return _http(self.responses[req.full_url], req.full_url)

    http_open = _serve
    https_open = _serve


def _opener_with(transport):
    return lambda *handlers: urllib.request.build_opener(*handlers, transport)


class TestTheRedirectGuardIsInThePath:
    """Acceptance 22, 23 and 28 through `download_ontology`, not beside it."""

    def _policy(self):
        return DestinationPolicy(
            resolver=lambda host: {"inside.example": [PRIVATE]}.get(host, [PUBLIC]),
            proxies={},
        )

    def test_a_redirect_to_ftp_is_refused_by_the_real_opener(self, tmp_path):
        transport = _CannedTransport({
            "https://purl.example/mondo.obo":
                b"HTTP/1.1 302 Found\r\nLocation: ftp://elsewhere.example/mondo.obo\r\n"
                b"Content-Length: 0\r\n\r\n",
        })

        with pytest.raises(OntologyDownloadError, match="scheme"):
            download_ontology("https://purl.example/mondo.obo", tmp_path / "m.obo",
                              policy=self._policy(), opener_factory=_opener_with(transport))

        assert not (tmp_path / "m.obo").exists()

    def test_a_redirect_into_the_network_is_refused_by_the_real_opener(self, tmp_path):
        transport = _CannedTransport({
            "https://purl.example/mondo.obo":
                b"HTTP/1.1 302 Found\r\nLocation: https://inside.example/mondo.obo\r\n"
                b"Content-Length: 0\r\n\r\n",
        })

        with pytest.raises(OntologyDownloadError, match=PRIVATE):
            download_ontology("https://purl.example/mondo.obo", tmp_path / "m.obo",
                              policy=self._policy(), opener_factory=_opener_with(transport))

        assert transport.opened == ["https://purl.example/mondo.obo"], (
            "the internal host was contacted before the refusal"
        )

    def test_an_ordinary_redirect_is_followed_by_the_real_opener(self, tmp_path):
        body = b"format-version: 1.2\n"
        transport = _CannedTransport({
            "https://purl.example/mondo.obo":
                b"HTTP/1.1 302 Found\r\nLocation: https://github.example/mondo.obo\r\n"
                b"Content-Length: 0\r\n\r\n",
            "https://github.example/mondo.obo":
                b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode()
                + b"\r\n\r\n" + body,
        })

        download_ontology("https://purl.example/mondo.obo", tmp_path / "m.obo",
                          policy=self._policy(), opener_factory=_opener_with(transport))

        assert (tmp_path / "m.obo").read_bytes() == body


class TestTheRequestGoesWhereTheRuleLooked:
    """The exemption for "no local answer" rests on one claim: the request goes
    to the proxy. So the rule's answer and the opener's route must agree, and
    that is checked against the opener `download_ontology` really builds —
    `ProxyHandler` included — rather than against a restatement of it."""

    @pytest.mark.parametrize("variables,url", [
        ({"http_proxy": "http://proxy.example:3128", "no_proxy": "unresolved.example:8080"},
         "http://unresolved.example:8080/m.obo"),
        ({"http_proxy": "http://proxy.example:3128", "no_proxy": "unresolved.example:8080"},
         "http://unresolved.example/m.obo"),
        ({"http_proxy": "http://proxy.example:3128", "no_proxy": "unresolved.example"},
         "http://unresolved.example:8080/m.obo"),
        ({"https_proxy": "http://proxy.example:3128", "no_proxy": ".example"},
         "https://purl.example/m.obo"),
        ({"https_proxy": "http://proxy.example:3128", "no_proxy": "*"},
         "https://purl.example/m.obo"),
        ({"https_proxy": "http://proxy.example:3128", "no_proxy": "PURL.EXAMPLE"},
         "https://purl.example/m.obo"),
        ({"https_proxy": "http://proxy.example:3128", "no_proxy": "other.test"},
         "https://purl.example/m.obo"),
        ({"https_proxy": "http://proxy.example:3128"}, "https://purl.example/m.obo"),
        ({"http_proxy": "http://proxy.example:3128"}, "https://purl.example/m.obo"),
    ], ids=["port-bypassed", "port-entry-other-port", "host-entry-any-port", "suffix",
            "wildcard", "case", "unrelated", "no-bypass", "other-scheme-only"])
    @pytest.mark.parametrize("mapping", [False, True], ids=["environment", "mapping"])
    def test_the_rule_and_the_opener_agree(self, tmp_path, monkeypatch, variables, url, mapping):
        _proxy_environment(monkeypatch, **variables)
        proxies = ({key.split("_")[0]: value for key, value in variables.items()
                    if key != "no_proxy"} if mapping else None)
        # A public answer, so the check passes either way and the request is
        # made: what is compared is only where it goes.
        policy = DestinationPolicy(resolver=lambda host: [PUBLIC], proxies=proxies)
        body = b"format-version: 1.2\n"
        hosts = []

        class Recording(_CannedTransport):
            def _serve(self, req):
                hosts.append(req.host)
                return _http(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode()
                             + b"\r\n\r\n" + body, req.full_url)

            http_open = _serve
            https_open = _serve

        download_ontology(url, tmp_path / "m.obo", policy=policy,
                          opener_factory=_opener_with(Recording({})))

        assert (hosts == ["proxy.example:3128"]) == policy.routes_through_proxy(url), hosts

    def test_the_opener_uses_the_proxies_the_rule_decided_on(self, tmp_path, monkeypatch):
        """The check deferred an unresolvable name to *this* proxy. A request
        that then went anywhere else — straight out, or to whatever the
        environment names — would be one the rule never judged."""
        for var in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
            monkeypatch.delenv(var, raising=False)
            monkeypatch.delenv(var.upper(), raising=False)
        body = b"format-version: 1.2\n"
        hosts = []

        class Recording(_CannedTransport):
            def _serve(self, req):
                hosts.append(req.host)
                return super()._serve(req)

            http_open = _serve
            https_open = _serve

        transport = Recording({
            "https://purl.example/mondo.obo":
                b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode()
                + b"\r\n\r\n" + body,
        })
        policy = DestinationPolicy(
            resolver=TestNoLocalAnswerIsNotAPolicyVerdict._unresolvable,
            proxies={"https": "http://proxy.example:3128"},
        )

        download_ontology("https://purl.example/mondo.obo", tmp_path / "m.obo",
                          policy=policy, opener_factory=_opener_with(transport))

        assert hosts == ["proxy.example:3128"]


_PROXY_VARIABLES = ("http_proxy", "https_proxy", "ftp_proxy", "all_proxy", "no_proxy")


def _proxy_environment(monkeypatch, **values):
    """Exactly these proxy settings, and none of the machine's own.

    This sandbox sets `HTTPS_PROXY` itself; a test of routing that inherited it
    would be a test of the sandbox."""
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


class TestNoLocalAnswerIsNotAPolicyVerdict:
    """The review's second P2. On a network whose only egress is a proxy,
    external names have no local answer; the lookup was typed as a refusal, so
    every source was refused and the operator told it was "a configuration
    decision". `urlretrieve` fetched through the same proxy without looking."""

    @pytest.fixture(autouse=True)
    def _no_inherited_proxies(self, monkeypatch):
        _proxy_environment(monkeypatch)

    @staticmethod
    def _unresolvable(host):
        raise _socket.gaierror(_socket.EAI_NONAME, "Name or service not known")

    def test_through_a_proxy_an_unresolvable_name_is_left_to_the_proxy(self):
        check_destination(
            "https://purl.obolibrary.org/obo/mondo.obo",
            DestinationPolicy(resolver=self._unresolvable,
                              proxies={"https": "http://proxy.hospital:3128"}),
        )

    def test_without_a_proxy_it_is_a_transfer_failure_not_a_refusal(self):
        from src.ontology.download import OntologyDestinationRefused, OntologyHostUnresolved

        with pytest.raises(OntologyHostUnresolved) as caught:
            check_destination("https://purl.obolibrary.org/obo/mondo.obo",
                              DestinationPolicy(resolver=self._unresolvable, proxies={}))

        assert not isinstance(caught.value, OntologyDestinationRefused)

    def test_a_host_the_proxy_is_bypassed_for_gets_no_exemption(self, monkeypatch):
        """Bypassed by `no_proxy`, which is where `ProxyHandler` reads it. (This
        test used to put `"no"` in the mapping — a list the opener never
        consults, so it asserted agreement with something that was not there.)"""
        from src.ontology.download import OntologyHostUnresolved

        _proxy_environment(monkeypatch, no_proxy="mirror.internal")
        with pytest.raises(OntologyHostUnresolved):
            check_destination(
                "https://mirror.internal/mondo.obo",
                DestinationPolicy(resolver=self._unresolvable,
                                  proxies={"https": "http://proxy:3128"}),
            )

    def test_a_bypass_written_with_a_port_is_honoured(self, tmp_path, monkeypatch):
        """The reviewer's reproduction, through the real opener. `no_proxy`
        naming `host:8080` sends `http://host:8080/` direct; the check used to
        judge the bare host name, call it proxied, and exempt it."""
        from src.ontology.download import OntologyHostUnresolved

        _proxy_environment(monkeypatch, http_proxy="http://proxy.example:3128",
                           no_proxy="unresolved.example:8080")
        transport = _CannedTransport({})

        with pytest.raises(OntologyHostUnresolved):
            download_ontology("http://unresolved.example:8080/mondo.obo", tmp_path / "m.obo",
                              policy=DestinationPolicy(resolver=self._unresolvable),
                              opener_factory=_opener_with(transport))

        assert transport.opened == [], "a request left although the rule refused it"

    def test_a_name_that_resolves_inside_is_refused_even_through_a_proxy(self):
        """The exemption is for no local answer, not for having a proxy."""
        with pytest.raises(OntologyDownloadError, match="globally routable"):
            check_destination(
                "https://sneaky.example/mondo.obo",
                DestinationPolicy(resolver=lambda h: [PRIVATE],
                                  proxies={"https": "http://proxy:3128"}),
            )

    def test_the_loader_reports_it_as_a_transfer_failure(self, tmp_path, monkeypatch):
        """With no proxy the fetch fails, and the operator gets the manual route
        rather than an instruction to edit `allowed_hosts`."""
        import src.ontology.settings as settings_module
        from src.ontology import download as download_module
        from src.ontology.download import OntologyHostUnresolved
        from src.ontology.loader import OntologyFetchError, OntologyLoader

        config = tmp_path / "c.yaml"
        config.write_text("ontology:\n  sources:\n    mondo:\n      - https://purl.example/mondo.obo\n")
        genuine = settings_module.load_ontology_settings
        monkeypatch.setattr(settings_module, "load_ontology_settings",
                            lambda config_path=None: genuine(config))

        def offline(url, target, **kwargs):
            raise OntologyHostUnresolved("purl.example could not be resolved here")

        monkeypatch.setattr(download_module, "download_ontology", offline)

        with pytest.raises(OntologyFetchError) as caught:
            OntologyLoader(cache_dir=tmp_path / "cache")._download_ontology("mondo", False)

        message = str(caught.value)
        assert "refused by the destination policy" not in message
        assert "manually" in message


class TestTheTransferEdges:

    def test_a_chunked_body_is_not_judged_by_a_content_length(self, tmp_path):
        """Transfer-Encoding overrides Content-Length; a complete chunked body
        compared with a length that does not describe it was refused."""
        payload = b"format-version: 1.2\n"
        raw = (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 500\r\n\r\n"
               + hex(len(payload))[2:].encode() + b"\r\n" + payload + b"\r\n0\r\n\r\n")

        class Opener:
            def open(self, url, timeout=None):
                return _http(raw, url)

        download_ontology("https://purl.example/m.obo", tmp_path / "m.obo",
                          policy=policy({"purl.example": [PUBLIC]}),
                          opener_factory=lambda *h: Opener())

        assert (tmp_path / "m.obo").read_bytes() == payload

    def test_an_interrupt_mid_transfer_leaves_no_partial_file(self, tmp_path):
        class Body:
            headers = {}

            def read(self, n):
                raise KeyboardInterrupt

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class Opener:
            def open(self, url, timeout=None):
                return Body()

        with pytest.raises(KeyboardInterrupt):
            download_ontology("https://purl.example/m.obo", tmp_path / "m.obo",
                              policy=policy({"purl.example": [PUBLIC]}),
                              opener_factory=lambda *h: Opener())

        assert list(tmp_path.iterdir()) == []

    def test_the_published_file_carries_the_umask_mode(self, tmp_path):
        """`NamedTemporaryFile` creates 0600; another account reading a shared
        root found the file unreadable and treated the root as empty."""
        import os
        import stat

        payload = b"format-version: 1.2\n"

        class Opener:
            def open(self, url, timeout=None):
                return _http(b"HTTP/1.1 200 OK\r\nContent-Length: "
                             + str(len(payload)).encode() + b"\r\n\r\n" + payload, url)

        target = download_ontology("https://purl.example/m.obo", tmp_path / "m.obo",
                                   policy=policy({"purl.example": [PUBLIC]}),
                                   opener_factory=lambda *h: Opener())
        mask = os.umask(0)
        os.umask(mask)

        assert stat.S_IMODE(target.stat().st_mode) == 0o666 & ~mask

    @pytest.mark.parametrize("url", ["http://[::1/mondo.obo", "http://host:notaport/mondo.obo"])
    def test_a_malformed_url_is_refused_not_leaked(self, url):
        with pytest.raises(OntologyDownloadError, match="well-formed"):
            check_destination(url, policy())

    def test_the_refusal_names_the_address(self):
        """Acceptance 25 promises the address in the refusal; it was never
        asserted."""
        with pytest.raises(OntologyDownloadError, match=PRIVATE):
            check_destination("https://mirror.internal/m.obo",
                              policy({"mirror.internal": [PRIVATE]}))


class TestTheOwlFallbackGoesThroughTheSameGate:
    """Acceptance 24, behaviourally. The previous test grepped the source for
    `urlretrieve`, which a new fetch path that is not called that would pass."""

    def test_both_attempts_reach_download_ontology_and_nothing_else(self, tmp_path, monkeypatch):
        import src.ontology.settings as settings_module
        from src.ontology import download as download_module
        from src.ontology.loader import OntologyLoader

        def forbidden(*a, **k):
            raise AssertionError("a fetch bypassed download_ontology")

        monkeypatch.setattr(urllib.request, "urlopen", forbidden)
        monkeypatch.setattr(urllib.request, "urlretrieve", forbidden)
        for name in ("socket", "create_connection", "getaddrinfo"):
            monkeypatch.setattr(_socket, name, forbidden, raising=False)

        config = tmp_path / "c.yaml"
        config.write_text("ontology:\n  sources:\n    mondo:\n"
                          "      - https://purl.example/mondo.obo\n"
                          "      - https://purl.example/mondo.owl\n")
        genuine = settings_module.load_ontology_settings
        monkeypatch.setattr(settings_module, "load_ontology_settings",
                            lambda config_path=None: genuine(config))

        seen = []

        def recorder(url, target, **kwargs):
            seen.append((url, Path(target).name))
            if url.endswith(".obo"):
                raise OntologyDownloadError("connection reset")
            Path(target).write_text("<rdf:RDF/>")
            return Path(target)

        monkeypatch.setattr(download_module, "download_ontology", recorder)

        cache = tmp_path / "cache"
        staged, final = OntologyLoader(cache_dir=cache)._download_ontology("mondo", False)

        assert [url for url, _ in seen] == ["https://purl.example/mondo.obo",
                                            "https://purl.example/mondo.owl"]
        for (_, name), prefix in zip(seen, (".mondo.obo.", ".mondo.owl.")):
            assert name.startswith(prefix) and name.endswith(".staged"), name
        assert staged.name == seen[1][1] and final == cache / "mondo.owl"
        assert [p.name for p in cache.iterdir()] == [staged.name], (
            "the failed OBO attempt left its staging file behind"
        )


class TestTheSettingsFindings:

    def test_a_source_listed_as_hp_is_the_hpo_source(self, tmp_path):
        """It was accepted, scheme-checked and never read: the loader asks for
        `hpo`, and the public PURL was fetched instead of the configured
        mirror."""
        config = tmp_path / "c.yaml"
        config.write_text("ontology:\n  sources:\n    hp:\n      - https://mirror.example/hp.obo\n")

        assert load_ontology_settings(config).urls_for("hpo") == ("https://mirror.example/hp.obo",)

    def test_a_malformed_source_url_is_a_settings_error(self, tmp_path):
        config = tmp_path / "c.yaml"
        config.write_text("ontology:\n  sources:\n    mondo:\n      - http://[::1/mondo.obo\n")

        with pytest.raises(OntologySettingsError, match="well-formed"):
            load_ontology_settings(config)

    @pytest.mark.parametrize("entry", ["https://mirror.internal", "mirror.internal:8443",
                                       "mirror.internal/onto", "user@mirror.internal"])
    def test_an_allow_list_entry_that_can_never_match_is_refused(self, tmp_path, entry):
        """The policy compares a bare host name, so these silently allowed
        nothing and the operator got refusals with no explanation."""
        config = tmp_path / "c.yaml"
        config.write_text(f"ontology:\n  allowed_hosts:\n    - \"{entry}\"\n")

        with pytest.raises(OntologySettingsError, match="bare host name"):
            load_ontology_settings(config)

    @pytest.mark.parametrize("entry,stored", [("Mirror.Internal", "mirror.internal"),
                                              ("10.0.0.5", "10.0.0.5"), ("[fd00::1]", "fd00::1")])
    def test_a_matchable_entry_is_kept(self, tmp_path, entry, stored):
        config = tmp_path / "c.yaml"
        config.write_text(f"ontology:\n  allowed_hosts:\n    - \"{entry}\"\n")

        assert load_ontology_settings(config).allowed_hosts == (stored,)
