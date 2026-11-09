"""
The run environment's named components and native build identity (E1).
=======================================================================
`src/utils/run_environment.py` (research-comparison README §5 Stage 2, §6 E1;
BACKLOG 71). The one collector records each named component in one shape:
installed, available, selected and observed in effect told apart, a fallback
with its reason, the distribution version, the native identity (a count and an
aggregate digest, never a file name) and the source receipt's facts only when
the receipt matches what is installed. Pinned here:

- every fact is named, and nothing names a path, a file or a credential;
- the aggregate is the specified digest, and two binaries of one version are
  told apart;
- installed or importable never reads as in effect; a fallback carries its
  reason;
- the receipt's facts are recorded only when its digests match, and every
  other case says why;
- the components are read once per process, and a fact that cannot be read is
  a status, never an error;
- an extra component's facts (a test double: no such component exists) reach
  a checkpoint's, a manifest's and a ledger record's environment unchanged,
  and the record view; no component enters a semantic digest or a condition
  group;
- the one projection and its one line of text.

The extensions here are stand-ins: distributions installed into this test's
own directory, the import probe's answers given, and the receipt's place this
test's, so no real package is imported, hashed or changed.

Module: tests/unit/test_environment_components.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils import pyg_native_check as native
from src.utils import run_environment
from src.utils.pyg_native_check import (
    NATIVE_EXTENSIONS,
    ExtensionStatus,
    aggregate_sha256,
    build_receipt,
    check_installed_against_wheels,
    describe_source,
    write_receipt,
)
from src.utils.run_environment import (
    COMPONENT_FIELDS,
    PYG_NATIVE_KIND,
    UNKNOWN,
    ComponentReader,
    collect_components,
    collect_environment,
    component_facts,
    native_identity,
    native_identity_text,
)
from tests.unit.test_pyg_install_identity import _install, _wheel

NATIVES = {"_scatter_cpu.so": b"scatter kernels", "lib/libcore.so.1": b"core library"}


def _natives(name, natives=NATIVES):
    return {f"{name}/{member}": data for member, data in natives.items()}


@pytest.fixture
def stack(tmp_path, monkeypatch):
    """PyG's native extensions as stand-ins: each installed from a wheel into
    this test's site directory, every one importable, no receipt kept; the
    collector reads afresh, and reads only these."""
    site = tmp_path / "venv" / "site"
    venv = tmp_path / "venv"
    statuses = {}
    wheels = {}

    def install(name, version="1.0", natives=None, available=True):
        directory = tmp_path / "wheels" / f"{name}-{version}-{len(wheels)}"
        directory.mkdir(parents=True)
        wheel = _wheel(directory, name, version, _natives(name, natives or NATIVES))
        _install(wheel, site)
        wheels[name] = wheel
        statuses[name] = ExtensionStatus(name, available, version if available else None,
                                         None if available else "broken")
        return wheel

    def unavailable(name):
        statuses[name] = ExtensionStatus(name, False, None, f"No module named '{name}'")

    real_identity, real_receipt = native.installed_identity, native.read_receipt
    monkeypatch.setattr(native, "installed_identity",
                        lambda name, path=None: real_identity(name, [str(site)]))
    monkeypatch.setattr(native, "check_pyg_native_extensions", lambda force=False: statuses)
    monkeypatch.setattr(native, "read_receipt", lambda prefix=None: real_receipt(venv))
    monkeypatch.setattr(run_environment, "_components_cache", None)
    for name in NATIVE_EXTENSIONS:
        install(name)

    def fresh():
        monkeypatch.setattr(run_environment, "_components_cache", None)
        return collect_components()

    return SimpleNamespace(site=site, venv=venv, install=install, unavailable=unavailable,
                           wheels=wheels, fresh=fresh, root=tmp_path)


def _digests(name, natives=NATIVES):
    return {member: hashlib.sha256(data).hexdigest() for member, data in _natives(name, natives).items()}


# --- Every fact named; no path, no file name ------------------------------------

def test_each_extension_is_a_component_with_every_fact_and_no_path_or_file_name(stack):
    environment = collect_environment("cpu")

    components = environment["components"]
    assert list(components) == list(NATIVE_EXTENSIONS)
    for name, facts in components.items():
        assert list(facts) == list(COMPONENT_FIELDS)
        assert facts["kind"] == PYG_NATIVE_KIND
        assert (facts["installed"], facts["available"], facts["version"]) == (True, True, "1.0")
        assert facts["native"] == {"state": "recorded", "count": 2,
                                   "aggregate_sha256": aggregate_sha256(_digests(name))}
    run_environment.assert_primitive(environment)
    text = json.dumps(environment)
    for private in (str(stack.root), "site-packages", "/home/", "_scatter_cpu", "libcore",
                    *(wheel.name for wheel in stack.wheels.values())):
        assert private not in text, private


def test_the_aggregate_is_one_sha256_over_sorted_name_and_digest_lines(stack):
    """The method the docstring specifies (`read_pyg_native`): one line
    ``<name>\\t<sha256>\\n`` per native library, its name relative to
    site-packages, in code-point order, hashed once; the names never recorded."""
    lines = "".join(f"{member}\t{digest}\n"
                    for member, digest in sorted(_digests("torch_scatter").items()))

    native_block = collect_components()["torch_scatter"]["native"]

    assert native_block["aggregate_sha256"] == hashlib.sha256(lines.encode()).hexdigest()
    assert native_block["count"] == 2


def test_two_binaries_of_one_version_are_told_apart(stack):
    first = stack.fresh()["torch_scatter"]
    (stack.site / "torch_scatter" / "_scatter_cpu.so").write_bytes(b"another build")
    second = stack.fresh()["torch_scatter"]

    assert first["version"] == second["version"] == "1.0"
    assert first["native"]["count"] == second["native"]["count"] == 2
    assert first["native"]["aggregate_sha256"] != second["native"]["aggregate_sha256"]
    one, other = (native_identity({"torch_scatter": facts}) for facts in (first, second))
    assert native_identity_text(one) != native_identity_text(other)


# --- Installed, available, selected, in effect and fallback, apart --------------

def test_installed_or_importable_never_reads_as_in_effect(stack):
    for facts in collect_components().values():
        assert (facts["installed"], facts["available"]) == (True, True)
        assert facts["selected"] == facts["in_effect"] == UNKNOWN
        assert facts["fallback_reason"] is None


def test_a_fallback_carries_its_reason(stack):
    stack.install("pyg_lib", available=False)
    stack.unavailable("torch_cluster")
    for path in stack.site.glob("torch_cluster*"):
        shutil.rmtree(path)

    components = stack.fresh()

    broken, absent = components["pyg_lib"], components["torch_cluster"]
    assert (broken["installed"], broken["available"], broken["in_effect"]) == (True, False, False)
    # Only what was observed: what PyG does instead differs by operation and
    # by release (2.8.1 ignores torch_cluster, and refuses some operations
    # without pyg_lib or torch_sparse), and nothing here observes it.
    assert broken["fallback_reason"] == "it is installed but does not import"
    assert broken["native"]["state"] == "recorded" and broken["version"] == "1.0"
    assert (absent["installed"], absent["available"], absent["in_effect"]) == (False, False, False)
    assert absent["fallback_reason"] == "it is not installed"
    assert absent["version"] is None
    assert absent["native"] == {"state": "not installed", "count": None, "aggregate_sha256": None}
    assert absent["source"] == {"state": "not recorded", "reason": "it is not installed"}
    assert broken["selected"] == absent["selected"] == UNKNOWN
    assert broken["requested"] == absent["requested"] == UNKNOWN


def test_an_extension_that_imports_without_a_distribution_is_not_said_to_be_absent(stack):
    """Something imports under the name, but no distribution's metadata names
    it (a copied directory, say): whether it is installed is not known, and
    nothing says it is not installed."""
    shutil.rmtree(stack.site / "torch_sparse-1.0.dist-info")

    facts = stack.fresh()["torch_sparse"]

    assert (facts["installed"], facts["available"], facts["in_effect"]) == (UNKNOWN, True,
                                                                             UNKNOWN)
    assert facts["fallback_reason"] is None and facts["version"] is None
    assert facts["native"] == {"state": "cannot be read", "count": None,
                               "aggregate_sha256": None, "reason": "it imports, but no "
                               "installed distribution lists its files"}
    assert facts["source"] == {"state": "not recorded",
                               "reason": "no installed distribution names it"}


_NATIVE = {"state": "recorded", "count": 2, "aggregate_sha256": "a" * 64}
_SOURCE = {"state": "recorded", "route": "pip install"}

#: Facts no observation can say, and so facts the record view would refuse
#: (each would hide the record that carries it): every one is refused when
#: it is collected, and becomes the component's status.
REFUSED = [
    ({"in_effect": True, "fallback_reason": "dtype"}, "in effect is available"),
    ({"in_effect": True, "available": False}, "in effect is available"),
    ({"selected": "yes"}, "selected is true, false or 'unknown'"),
    ({"requested": "flash"}, "requested is true, false or 'unknown'"),
    ({"installed": 1}, "installed is true, false or 'unknown'"),
    ({"fallback_reason": ""}, "fallback reason is text"),
    ({"version": 2}, "version is text"),
    ({"native": {"count": 1}}, "native identity has one of the states"),
    ({"native": {"state": "bundled", "count": None, "aggregate_sha256": None}},
     "native identity has one of the states"),
    ({"native": {"state": "not recorded"}}, "gives its count and aggregate_sha256"),
    ({"native": {**_NATIVE, "aggregate_sha256": "abc"}}, "64 lowercase hex"),
    ({"native": {**_NATIVE, "aggregate_sha256": "A" * 64}}, "64 lowercase hex"),
    ({"native": {**_NATIVE, "count": -1}}, "whole number of at least 0"),
    ({"native": {**_NATIVE, "count": True}}, "whole number of at least 0"),
    ({"native": {**_NATIVE, "count": "one"}}, "whole number of at least 0"),
    ({"native": {**_NATIVE, "count": None}}, "given exactly when it is recorded"),
    ({"native": {**_NATIVE, "aggregate_sha256": None}}, "given exactly when it is recorded"),
    ({"native": {"state": "cannot be read", "count": 2, "aggregate_sha256": "a" * 64}},
     "given exactly when it is recorded"),
    ({"native": {"state": "not installed", "count": None, "aggregate_sha256": None,
                 "reason": ""}}, "native identity reason is text"),
    ({"source": "receipt"}, "source is 'recorded' or 'not recorded'"),
    ({"source": {"state": "bundled with torch"}}, "source is 'recorded' or 'not recorded'"),
    ({"source": {"state": "observed at run time"}}, "source is 'recorded' or 'not recorded'"),
    ({"source": {"state": "recorded"}}, "route, as text, exactly when it is recorded"),
    ({"source": {"state": "recorded", "route": ""}}, "route, as text, exactly when"),
    ({"source": {"state": "recorded", "route": 3}}, "route, as text, exactly when"),
    ({"source": {"state": "not recorded", "route": "pip install"}},
     "route, as text, exactly when it is recorded"),
    ({"source": {"state": "not recorded", "reason": 5}}, "source reason is text"),
    ({"source": {**_SOURCE, "flags": {"MAX_JOBS": float("inf")}}}, "Out of range float"),
    ({"source": {**_SOURCE, "flags": {"MAX_JOBS": float("nan")}}}, "Out of range float"),
    ({"speed": "fast"}, "outside a component's shape"),
]


@pytest.mark.parametrize("facts, words", REFUSED)
def test_component_facts_refuses_what_no_observation_can_say(facts, words):
    with pytest.raises(ValueError, match=words):
        component_facts("attention kernel", **facts)


#: Facts a reader may give, among them a component installed by pip with a
#: recorded source and no native library, and one whose libraries could not
#: all be read.
ACCEPTED = [
    {},
    {"requested": True, "selected": False, "available": True, "in_effect": False,
     "fallback_reason": "the setting asks for flash, the dispatcher chose math"},
    {"installed": True, "version": "2.0", "native": _NATIVE,
     "source": {**_SOURCE, "requested": {"kind": "registry", "name": "x", "specifier": ""},
                "flags": {"MAX_JOBS": 8, "SCALE": 1.5}}},
    {"native": {"state": "recorded", "count": 0,
                "aggregate_sha256": hashlib.sha256(b"").hexdigest()}, "source": _SOURCE},
    {"native": {"state": "cannot be read", "count": 3, "aggregate_sha256": None,
                "reason": "a library cannot be read"}},
    {"native": {"state": "not installed", "count": None, "aggregate_sha256": None},
     "source": {"state": "not recorded", "reason": "it is not installed"}},
]


@pytest.mark.parametrize("facts", ACCEPTED)
def test_every_component_the_collector_accepts_the_record_view_reads(facts):
    """One shape on both sides of the record (E1): whatever `component_facts`
    accepts, the record view validates (`RecordComponent`), so no record is
    hidden for the environment it was measured in (§4.2)."""
    from src.api.routes.evaluations import RecordComponent

    collected = component_facts("attention kernel", **facts)

    (shown,) = native_identity({"test_attention": collected}).values()
    RecordComponent.model_validate(shown)


def test_a_reader_giving_what_the_view_would_refuse_becomes_a_readable_status(
        stack, monkeypatch):
    """A second reader that gives a recorded source without its route (or any
    other refused fact) is recorded as that component's status, which the
    record view reads; the reader's other components are kept."""
    from src.api.routes.evaluations import RecordComponent

    reader = ComponentReader("attention kernel", ("test_attention", "test_kernel"),
                             lambda names: {
                                 "test_attention": {"installed": True,
                                                    "source": {"state": "recorded"}},
                                 "test_kernel": {"installed": True, "source": _SOURCE}})
    monkeypatch.setattr(run_environment, "COMPONENT_READERS",
                        (*run_environment.COMPONENT_READERS, reader))

    components = stack.fresh()

    assert components["test_attention"]["installed"] == UNKNOWN
    assert components["test_attention"]["source"] == {"state": "not recorded",
                                                      "reason": "its facts could not be read"}
    assert components["test_kernel"]["source"] == _SOURCE
    for shown in native_identity(components).values():
        RecordComponent.model_validate(shown)


def test_component_facts_reads_what_is_not_observed_as_not_observed():
    facts = component_facts("attention kernel", installed=True)

    assert facts == {
        "kind": "attention kernel", "requested": UNKNOWN, "installed": True,
        "available": UNKNOWN, "selected": UNKNOWN, "in_effect": UNKNOWN,
        "fallback_reason": None, "version": None,
        "native": {"state": "not recorded", "count": None, "aggregate_sha256": None},
        "source": {"state": "not recorded", "reason": "no source is recorded for this component"}}
    with pytest.raises(TypeError):
        component_facts("attention kernel", version="1", source={**_SOURCE, "flags": (1, 2)})


# --- The source receipt: recorded only where it matches -------------------------

def _keep_receipt(stack, name="torch_scatter"):
    checks = check_installed_against_wheels([stack.wheels[name]], path=[str(stack.site)])
    wheel = stack.wheels[name]
    receipt = build_receipt(
        route="local build", checks=checks,
        requested={wheel.name: describe_source(
            "git+https://reader:t0ken@github.com/rusty1s/pytorch_scatter.git@2.1.2")},
        commits={wheel.name: "c" * 40},
        compiled_against={"torch": "2.10.0+cu130", "cuda": "13.0",
                          "basis": "the build's target environment"},
        toolchain={"nvcc": "13.0.88", "gcc": "13.3.0", "g++": "13.3.0"},
        flags={"TORCH_CUDA_ARCH_LIST": "12.1", "FORCE_CUDA": "1", "MAX_JOBS": "8"})
    write_receipt(receipt, stack.venv)
    return receipt


def test_the_receipts_facts_are_recorded_when_its_digests_match(stack):
    _keep_receipt(stack)
    wheel = stack.wheels["torch_scatter"]

    components = stack.fresh()

    assert components["torch_scatter"]["source"] == {
        "state": "recorded", "route": "local build",
        "requested": {"kind": "vcs", "vcs": "git", "ref": "2.1.2",
                      "url": "https://github.com/rusty1s/pytorch_scatter.git"},
        "resolved": {"version": "1.0", "commit": "c" * 40},
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "compiled_against": {"torch": "2.10.0+cu130", "cuda": "13.0",
                             "basis": "the build's target environment"},
        "toolchain": {"nvcc": "13.0.88", "gcc": "13.3.0", "g++": "13.3.0"},
        "flags": {"TORCH_CUDA_ARCH_LIST": "12.1", "FORCE_CUDA": "1", "MAX_JOBS": "8"}}
    assert components["pyg_lib"]["source"] == {
        "state": "not recorded", "reason": "the source receipt does not name it"}
    text = json.dumps(components)
    for private in ("t0ken", "reader", wheel.name, str(stack.root)):
        assert private not in text
    identity = native_identity(components)
    assert identity["torch_scatter"]["source"] == {"state": "recorded", "route": "local build"}
    assert identity["pyg_lib"]["source"] == {"state": "not recorded"}


#: The prebuilt release's asset, as deploy.sh names it: a file name.
ASSET = "pyg-ext-torch2.10.0-cu130-cp312-linux_aarch64.tar.gz"


def test_a_prebuilt_release_receipt_is_recorded_without_its_asset_file_name(stack):
    """The prebuilt route's requested source names the release's asset file;
    the record keeps the tag and the asset's SHA-256, which names it, as the
    wheel's SHA-256 names the wheel."""
    checks = check_installed_against_wheels([stack.wheels["pyg_lib"]], path=[str(stack.site)])
    wheel = stack.wheels["pyg_lib"]
    release = {"kind": "prebuilt release", "tag": "pyg-ext-v1", "asset": ASSET,
               "asset_sha256": "d" * 64}
    write_receipt(build_receipt(
        route="prebuilt release", checks=checks, requested={wheel.name: release},
        compiled_against={"torch": "2.10.0", "cuda": "13.0",
                          "basis": "the release's declared target"},
        toolchain=None, flags=None), stack.venv)

    source = stack.fresh()["pyg_lib"]["source"]

    assert source == {
        "state": "recorded", "route": "prebuilt release",
        "requested": {"kind": "prebuilt release", "tag": "pyg-ext-v1", "asset_sha256": "d" * 64},
        "resolved": {"version": "1.0"},
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "compiled_against": {"torch": "2.10.0", "cuda": "13.0",
                             "basis": "the release's declared target"},
        "toolchain": None, "flags": None}
    assert ASSET not in json.dumps(source) and wheel.name not in json.dumps(source)


def _edit_receipt(stack, change):
    target = native.receipt_path(stack.venv)
    receipt = json.loads(target.read_text())
    change(receipt["wheels"][0])
    write_receipt(receipt, stack.venv)


def test_a_hand_edited_receipt_never_carries_a_credential_into_a_record(stack):
    """The deploy layer strips a source URL (`describe_source`); a receipt
    edited by hand whose digests still match is stripped again when it is
    copied (`strip_credentials`), and a requested source that is not an
    object is not copied at all."""
    _keep_receipt(stack)
    _edit_receipt(stack, lambda entry: entry.update(requested={
        "kind": "url", "url": "https://reader:t0ken@example.org/scatter.tar.gz?token=s3cret#x"}))

    source = stack.fresh()["torch_scatter"]["source"]

    assert source["state"] == "recorded"
    assert source["requested"] == {"kind": "url", "url": "https://example.org/scatter.tar.gz"}
    for secret in ("t0ken", "reader", "s3cret"):
        assert secret not in json.dumps(source)

    _edit_receipt(stack, lambda entry: entry.update(
        requested="git+https://reader:t0ken@github.com/rusty1s/pytorch_scatter.git"))
    assert stack.fresh()["torch_scatter"]["source"] == {
        "state": "not recorded",
        "reason": "the source receipt's requested source is not one this build reads"}


def test_a_receipt_that_does_not_match_records_no_source_and_says_why(stack):
    _keep_receipt(stack)
    (stack.site / "torch_scatter" / "_scatter_cpu.so").write_bytes(b"replaced since")

    source = stack.fresh()["torch_scatter"]["source"]

    assert source == {"state": "not recorded", "reason": (
        "the source receipt does not describe what is installed: its installed native "
        "libraries are not the ones the receipt records")}
    shutil.rmtree(stack.site / "torch_scatter-1.0.dist-info")
    stack.install("torch_scatter", version="2.0")
    assert stack.fresh()["torch_scatter"]["source"]["reason"].endswith(
        "installed version 2.0, the receipt's 1.0")


@pytest.mark.parametrize("content, reason", [
    (None, "no source receipt is kept in this environment"),
    (b"{not json", "the file in the source receipt's place is not a receipt this build reads"),
    (b'{"receipt": "shepherd-pyg-native-receipt", "schema": 1, "wheels": [], "x": NaN}',
     "the file in the source receipt's place is not a receipt this build reads"),
    (b'{"receipt": "shepherd-pyg-native-receipt", "schema": 1, "wheels": [], '
     b'"flags": {"MAX_JOBS": 1e400}}',
     "the file in the source receipt's place is not a receipt this build reads"),
])
def test_without_a_readable_receipt_each_source_says_why(stack, content, reason):
    if content is not None:
        target = native.receipt_path(stack.venv)
        target.parent.mkdir(parents=True)
        target.write_bytes(content)

    for facts in stack.fresh().values():
        assert facts["source"] == {"state": "not recorded", "reason": reason}


# --- Read once per process; a fact that cannot be read is a status --------------

def test_the_components_are_read_once_per_process_and_given_as_copies(stack, monkeypatch):
    reads = []
    real = native.installed_identity
    monkeypatch.setattr(native, "installed_identity",
                        lambda name, path=None: reads.append(name) or real(name))

    first = collect_environment("cpu")["components"]
    first["pyg_lib"]["version"] = "changed by a caller"
    second = collect_environment("cuda")["components"]

    assert reads == list(NATIVE_EXTENSIONS)
    assert second == collect_components() and second["pyg_lib"]["version"] == "1.0"


def test_a_fact_that_cannot_be_read_is_a_status_never_an_error(stack, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("/srv/private/site-packages cannot be read")

    monkeypatch.setattr(native, "installed_identity", broken)
    noisy = ComponentReader("attention kernel", ("test_attention",),
                            lambda names: {"test_attention": {"in_effect": True,
                                                              "fallback_reason": "dtype"}})
    failing = ComponentReader("sampler", ("test_sampler",),
                              lambda names: (_ for _ in ()).throw(RuntimeError("/srv/x")))
    monkeypatch.setattr(run_environment, "COMPONENT_READERS",
                        (*run_environment.COMPONENT_READERS, noisy, failing))

    components = stack.fresh()

    assert list(components) == [*NATIVE_EXTENSIONS, "test_attention", "test_sampler"]
    for facts in components.values():
        assert facts["installed"] == facts["in_effect"] == UNKNOWN
        assert facts["native"] == {"state": "cannot be read", "count": None,
                                   "aggregate_sha256": None,
                                   "reason": "its facts could not be read"}
        assert facts["source"] == {"state": "not recorded",
                                   "reason": "its facts could not be read"}
    assert "/srv" not in json.dumps(components)


# --- An extra component reaches every record unchanged ----------------------------

def _extra_reader():
    return ComponentReader("attention kernel", ("test_attention",), lambda names: {
        "test_attention": {"installed": True, "available": True, "selected": True,
                           "in_effect": False, "version": "2.0",
                           "fallback_reason": "the run's dtype is not supported"}})


def test_an_extra_components_facts_reach_the_checkpoint_the_manifest_and_the_ledger(
        tmp_path, monkeypatch):
    """E1 with a test double: a component nothing in the stack has, beside the
    real PyG extensions, through the production writers. The checkpoint's
    environment, the kept report's manifest and the ledger record carry its
    facts as the collector gave them, and the record view shows its identity;
    its fallback keeps its reason, and it never reads as in effect."""
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import measure_scorer

    from src.api.routes.evaluations import read_records
    from src.evaluation.sidecar import read_ledger
    from tests.fixtures.registered_cohort import (
        LABEL,
        graph_source_cases,
        import_version,
        write_demo_workspace,
        write_source,
    )
    from tests.fixtures.trained_children import train_children

    monkeypatch.setattr(run_environment, "COMPONENT_READERS",
                        (*run_environment.COMPONENT_READERS, _extra_reader()))
    monkeypatch.setattr(run_environment, "_components_cache", None)
    data_dir = write_demo_workspace(tmp_path / "workspace")
    registry = tmp_path / "registry"
    registry.mkdir()
    import_version(registry, data_dir, write_source(tmp_path / "cases.json",
                                                     graph_source_cases(data_dir)))
    trained = train_children(data_dir, ("disease",), runs=tmp_path / "runs")

    expected = collect_components()
    assert list(expected) == [*NATIVE_EXTENSIONS, "test_attention"]
    assert expected["test_attention"] == component_facts(
        "attention kernel", installed=True, available=True, selected=True, in_effect=False,
        version="2.0", fallback_reason="the run's dtype is not supported")
    for checkpoint in (trained["base"], trained["disease"]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["environment"]["components"] == expected

    monkeypatch.setenv("SHEPHERD_REGISTRY", str(registry))
    code = measure_scorer.main(["--mode", "disease", "--checkpoint", str(trained["disease"]),
                                "--data-dir", str(data_dir), "--cohort", LABEL,
                                "--registry", str(registry), "--device", "cpu", "--record"])
    assert code == 0
    (kept,) = registry.glob("versions/*/reports/*/report.json")
    manifest = json.loads(kept.read_text())["manifest"]
    assert manifest["environment"]["facts"]["components"] == expected
    ledger, _ = read_ledger(trained["disease"].parent / "evaluations.json")
    (record,) = ledger["records"]
    assert record["environment"] == manifest["environment"]

    viewed, unreadable = read_records(ledger, record["checkpoint_digest"])
    assert unreadable == []
    (_record, view), = viewed
    shown = view.model_dump(mode="json")["native_identity"]
    assert set(shown) == set(expected)  # the ledger's encoding orders keys
    assert shown["test_attention"] == {
        "kind": "attention kernel", "version": "2.0",
        "native": {"state": "not recorded", "count": None, "aggregate_sha256": None},
        "source": {"state": "not recorded", "route": None}}


def test_no_component_enters_a_semantic_digest_or_a_condition_group():
    """Recorded, never compared (R11): two reports that differ only in their
    components have one record key and the same semantic digests."""
    from src.evaluation.sidecar import build_record, record_key
    from tests.unit.test_evaluation_sidecar import environment_block, registered_report

    def report(components):
        environment = environment_block()
        environment["facts"]["components"] = components
        return registered_report(environment=environment)

    one = build_record(report({"pyg_lib": component_facts(PYG_NATIVE_KIND, version="0.6.0")}),
                       "1" * 64)
    other = build_record(report({"pyg_lib": component_facts(PYG_NATIVE_KIND, version="0.8.0"),
                                 "test_attention": component_facts("attention kernel")}),
                         "2" * 64)

    assert one["environment"] != other["environment"]
    assert record_key(one) == record_key(other)
    for digest in ("measurement_semantics_digest", "scoring_semantics_digest"):
        assert one[digest] == other[digest]


# --- The one projection and its one line --------------------------------------------

def _entry(version, digest, route=None, state="recorded", count=2):
    source = ({"state": "recorded", "route": route} if route
              else {"state": "not recorded", "reason": "no source receipt is kept"})
    return component_facts(PYG_NATIVE_KIND, installed=state != "not installed", version=version,
                           native={"state": state, "count": count if state != "not installed"
                                   else None,
                                   "aggregate_sha256": digest if state == "recorded" else None},
                           source=source)


def test_the_one_line_says_each_identity_and_one_shared_route_once():
    shared = native_identity({"pyg_lib": _entry("0.8.0", "a" * 64, "local build"),
                              "torch_scatter": _entry("2.1.2", "b" * 64, "local build"),
                              "torch_cluster": _entry(None, None, state="not installed")})
    assert native_identity_text(shared) == (
        "local build — pyg_lib 0.8.0 aaaaaaaaaaaa, torch_cluster not installed, "
        "torch_scatter 2.1.2 bbbbbbbbbbbb")
    # By name, whatever order the record keeps: a report keeps the
    # collector's, the ledger's encoding sorts keys; both read alike.
    assert native_identity_text(json.loads(json.dumps(shared, sort_keys=True))) == \
        native_identity_text(dict(reversed(list(shared.items())))) == \
        native_identity_text(shared)

    mixed = native_identity({"pyg_lib": _entry("0.8.0", "a" * 64, "local build"),
                             "torch_scatter": _entry("2.1.2", "b" * 64),
                             "torch_sparse": _entry("0.6.18", None, state="cannot be read")})
    assert native_identity_text(mixed) == (
        "pyg_lib 0.8.0 aaaaaaaaaaaa (local build), torch_scatter 2.1.2 bbbbbbbbbbbb (route not "
        "recorded), torch_sparse 0.6.18 libraries cannot be read (route not recorded)")

    assert native_identity_text(native_identity(None)) == "not recorded"
    assert native_identity_text({}) == "no component recorded"
    assert native_identity_text(["pyg_lib"]) == "cannot be read"
    assert native_identity_text({"x": "y"}) == "x cannot be read"
    assert native_identity_text(native_identity(
        {"pure": _entry("1.0", hashlib.sha256(b"").hexdigest(), "local build", count=0)})) == (
        "local build — pure 1.0 no native library")


def test_the_projection_picks_the_identity_and_passes_anything_else_on():
    facts = _entry("0.8.0", "a" * 64, "local build")
    facts["source"]["toolchain"] = {"gcc": "13.3.0"}

    assert native_identity({"pyg_lib": facts}) == {"pyg_lib": {
        "kind": PYG_NATIVE_KIND, "version": "0.8.0",
        "native": {"state": "recorded", "count": 2, "aggregate_sha256": "a" * 64},
        "source": {"state": "recorded", "route": "local build"}}}
    assert native_identity(["not", "an", "object"]) == ["not", "an", "object"]
    assert native_identity({"pyg_lib": {"version": "1"}}) == {"pyg_lib": {"version": "1"}}


def test_the_readers_project_records_and_never_collect():
    """The API's record view and the two pages show what a record holds: none
    of them reaches the collector or the install identity, which hash files
    (once per process, in training, measurement and the probe's A1)."""
    import ast

    collecting = {"collect_components", "collect_environment", "read_pyg_native",
                  "installed_identity", "installed_native_facts", "read_receipt"}
    repo = Path(__file__).resolve().parents[2]
    for module in ("src/api/routes/evaluations.py", "src/webui/components/test_page.py",
                   "src/webui/components/models_page.py"):
        tree = ast.parse((repo / module).read_text(encoding="utf-8"))
        named = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        named |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        named |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                  for alias in node.names}
        assert not named & collecting, module
