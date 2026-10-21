"""
Records come only from the readers: the static pin (contract M2.1, S9).
=======================================================================
M2.1 makes every recorded input's digest the digest of the bytes a reader parsed.
The dynamic pins run each entry point with every path hash poisoned and check
that no poison reaches its record (`test_training_reads.py`,
`test_measurement_reads_once.py`, `test_serving_reads_once.py`,
`test_sp_artifact_binding.py` for the SP producer, `test_evidence_scripts.py` and
`test_audit_generator_fidelity.py` for the audits S9 moved). This is the static
half: it reads the code instead of running it, so it also sees a path hashed in
a way the poison cannot reach.

**What it scans.** `src/**`, the entry points (`train_model.py`,
`measure_scorer.py`, `compute_shortest_paths.py`) and the scripts S9 moved onto
the readers (`probe_deployment.py`, `audit_split_overlap.py`,
`audit_generator_fidelity.py`). Item 19's other scripts and `build_index.py`
are outside it, by the plan (S9; BACKLOG item 19).

**What it fails on.**
- The path-form verifiers and `compute_input_digests`, deleted in S9, named
  anywhere in scope: there are not two permanent options.
- Any reference to a primitive that hashes or reads a file (`file_sha256`,
  `read_once`, `hashlib.*`, `torch.load`, `json.load(s)`,
  `KnowledgeGraph.load_json`) at a site not listed in `ALLOWED` below, whether it
  is called, passed as a value, re-exported or imported under another name, and
  any `ALLOWED` entry that no longer matches a site, so the list cannot go stale.
- A name bound by `from hashlib import file_digest`, which the dynamic pins'
  poison could not reach.

**Not "no path hash anywhere".** Each entry in `ALLOWED` is a named exception
with its use and its owner: the readers themselves, the three stated exceptions
and the `kg.json` identification check, item 18's and item 19's deferred sites
with their homes, and loads of files that are not provenance inputs at all
(configuration, UI state). Adding a site means adding it here with a reason, in
review; it is never a line added to make the test pass.

Module: tests/unit/test_provenance_pins.py
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

ENTRY_POINTS = ("scripts/train_model.py", "scripts/measure_scorer.py",
                "scripts/compute_shortest_paths.py")
MOVED_IN_S9 = ("scripts/probe_deployment.py", "scripts/audit_split_overlap.py",
               "scripts/audit_generator_fidelity.py")
#: M4's CLIs. They read and hash nothing themselves: the registry, the samples reader
#: and the registration function do, and each of those sites is listed below.
M4_SCRIPTS = ("scripts/import_cohort.py", "scripts/record_evaluation.py")

#: Deleted in S9. Named nowhere in scope, so neither a caller nor a copy survives.
REMOVED = ("verify_graph_artifacts", "verify_generated_cohorts", "verify_graph_source",
           "compute_input_digests")

_READER = "an M2.1 reader: parses the bytes of its one read_once and returns their identity"
_NOT_INPUT = ("not a provenance input: {what}; nothing recorded or compared is taken from it. "
              "Permanent, no home")
_IN_MEMORY = "a digest of {what} in memory, not of a file. Permanent, no home"
_PROBE = ("owner: the deployment probe's own evidence (scripts/probe_deployment.py, M1's "
          "acceptance, M2.4's D2 and M2's acceptance). ")

#: (file, enclosing function, primitive) -> (number of calls, use and owner).
ALLOWED = {
    # --- The readers and the primitives (contract M2.1, S1-S4) -------------
    ("src/utils/fingerprint.py", "read_once", "hashlib.sha256"):
        (1, "the primitive: the digest of the bytes it returns"),
    ("src/utils/fingerprint.py", "file_sha256", "hashlib.file_digest"):
        (1, "the primitive for a file nobody is about to parse; its callers are listed here"),
    ("src/kg/storage/file_storage.py", "_read_json", "json.loads"): (1, _READER),
    ("src/kg/storage/file_storage.py", "_read_json", "read_once"): (1, _READER),
    ("src/kg/storage/file_storage.py", "_read_tensors", "torch.load"): (1, _READER),
    ("src/kg/storage/file_storage.py", "_read_tensors", "read_once"): (1, _READER),
    ("src/kg/artifacts.py", "read_split_manifest", "json.loads"): (1, _READER),
    ("src/kg/artifacts.py", "read_split_manifest", "read_once"): (1, _READER),
    # Contract M3b: the dataset record, moved onto M2.1's reads (BACKLOG 19.14,
    # 19.15; the old status readers are gone).
    ("src/kg/artifacts.py", "verify_dataset_record", "json.loads"): (1, _READER),
    ("src/kg/artifacts.py", "verify_dataset_record", "read_once"): (1, _READER),
    # M5-1: the build record beside the manifest, descriptive and never an input;
    # read once and parsed from those bytes like the readers above.
    ("src/kg/artifacts.py", "read_workspace_build", "json.loads"): (1, _READER),
    ("src/kg/artifacts.py", "read_workspace_build", "read_once"): (1, _READER),
    # Contract M3b: the writers' digests are of the bytes they write (BACKLOG
    # 19.16, 19.17); every writer that binds a digest writes through this.
    ("src/utils/fingerprint.py", "DigestingFile.__init__", "hashlib.sha256"):
        (1, _IN_MEMORY.format(what="every byte a writer writes, as it is written (contract "
                                   "M3b)")),
    ("src/kg/graph.py", "KnowledgeGraph.read_json", "json.loads"): (1, _READER),
    ("src/kg/graph.py", "KnowledgeGraph.read_json", "read_once"): (1, _READER),
    ("src/utils/checkpoint_io.py", "read_checkpoint", "torch.load"): (1, _READER),
    ("src/utils/checkpoint_io.py", "read_checkpoint", "read_once"): (1, _READER),
    ("src/utils/checkpoint_io.py", "write_checkpoint", "hashlib.sha256"):
        (1, _IN_MEMORY.format(what="the checkpoint bytes about to be written (contract M3a-1)")),
    ("scripts/audit_generator_fidelity.py", "build_report", "read_once"):
        (1, "S9: the fidelity audit reads phenotype.hpoa once and counts its rows from those "
            "bytes; the digest it reports is that read's"),
    # --- Stated exceptions and decisions (M2.1) ------------------------------
    ("src/kg/artifacts.py", "check_unparsed_kg_json", "file_sha256"):
        (1, "M2.1 decision 1: identifies kg.json before a run that does not parse it, "
            "against the run's one manifest reading; records nothing"),
    ("src/inference/pipeline.py", "DiagnosisPipeline._load_shortest_paths", "torch.load"):
        (1, "M2.1 stated exception: the SP table at serving; its identity is the build_id "
            "inside the loaded tensor, checked against the sidecar"),
    ("src/inference/sp_artifact.py", "read_sidecar", "json.loads"):
        (1, "M2.1 stated exception: the SP sidecar's single read, parsed from the bytes it keeps"),
    ("scripts/compute_shortest_paths.py", "main", "file_sha256"):
        (1, "S9 allowlist: the SP producer's end-of-run comparison, a warning that records "
            "nothing; the sidecar records the digest of the bytes traversed"),
    # --- The evaluation ledger and registration (BACKLOG 18 done in M4) ------
    ("src/evaluation/sidecar.py", "checkpoint_matches", "file_sha256"):
        (1, "BACKLOG 19.7: identifies a checkpoint nobody is about to parse (M2.1's own "
            "allowance for file_sha256); the one comparison find_checkpoint makes for each "
            "candidate and register_report for the file record_evaluation --checkpoint "
            "names; records nothing"),
    ("src/evaluation/sidecar.py", "ledger_digest", "file_sha256"):
        (1, "the ledger's stale-write guard: compares the file by path at write time with "
            "the digest read_ledger returned for the bytes it parsed (BACKLOG item 18 done); "
            "identifies nothing parsed; its window to the rename stays open — single-writer, "
            "not compare-and-swap"),
    ("src/evaluation/sidecar.py", "read_ledger", "read_once"): (1, _READER),
    ("src/evaluation/sidecar.py", "read_ledger", "json.loads"): (1, _READER),
    # The one reader of a kept report (moved from registration.py in M5-1): the
    # registration, the report store's D5 scan and report_state read through it.
    ("src/evaluation/report_store.py", "read_report", "read_once"): (1, _READER),
    ("src/evaluation/report_store.py", "read_report", "json.loads"): (1, _READER),
    ("src/evaluation/report_store.py", "_read_entries", "read_once"):
        (1, _NOT_INPUT.format(what="a report store's usage history or investigation notes, "
                                   "append-only, read for display (M5-1, D4/D5); no record "
                                   "or comparison is taken from them")),
    ("src/evaluation/report_store.py", "_read_entries", "json.loads"):
        (1, _NOT_INPUT.format(what="one line of a report store's usage history or "
                                   "investigation notes, parsed from the bytes of its one "
                                   "read_once")),
    # --- The cohort registry (contract M4) ------------------------------------
    ("src/evaluation/cohort_registry.py", "prepare_import", "read_once"):
        (1, "M4: the cohort source file, parsed by parse_source and kept verbatim from the "
            "bytes of this one read; its digest names the kept copy and the version manifest"),
    ("src/evaluation/cohort_registry.py", "parse_source", "json.loads"):
        (1, "M4: parses the source bytes prepare_import read once; opens no file itself"),
    ("src/evaluation/cohort_registry.py", "_keep_source", "read_once"):
        (1, "M4: the no-clobber check of a kept source: an existing sources/<sha256>.json "
            "must hash to its name before an import relies on it; records nothing"),
    ("src/evaluation/cohort_registry.py", "resolve_version", "read_once"): (1, _READER),
    ("src/evaluation/cohort_registry.py", "resolve_version", "json.loads"): (1, _READER),
    ("src/ontology/loader.py", "OntologyLoader.load", "hashlib.sha256"):
        (1, "KG build, unchanged by M2.1 (BACKLOG 19.13): the digest of the handle that "
            "is parsed, M2.1's precedent"),
    ("src/ontology/resolver.py", "scan_identity", "hashlib.sha256"):
        (1, "KG build, unchanged by M2.1: one pass over an ontology file feeds both its "
            "digest and its identity scan"),
    # --- Digests of values in memory, not of files ---------------------------
    ("src/evaluation/sidecar.py", "_canonical_digest", "hashlib.sha256"):
        (1, _IN_MEMORY.format(what="a canonical JSON string of manifest fields; the "
                                   "semantics and scoring-semantics digests share it")),
    ("src/evaluation/cohort_registry.py", "_sha256", "hashlib.sha256"):
        (1, _IN_MEMORY.format(what="the mapped samples about to be written and the canonical "
                                   "form of a would-be version manifest (the preview id)")),
    ("src/utils/file_modes.py", "publish_bytes", "hashlib.sha256"):
        (1, _IN_MEMORY.format(what="the bytes publish_bytes is about to write, returned as "
                                   "the writer's digest of what it wrote (contract M4)")),
    ("src/kg/disease_allocation.py", "disease_set_digest", "hashlib.sha256"):
        (1, _IN_MEMORY.format(what="a disease set")),
    ("src/kg/disease_allocation.py", "universe_digest", "hashlib.sha256"):
        (1, _IN_MEMORY.format(what="a disease universe")),
    ("src/kg/sample_generator.py", "_sha256_text", "hashlib.sha256"):
        (1, "the generator's digest of the text it writes, equal to the file's by "
            "construction (the writer's digest of what it wrote, as contract M3b "
            "requires of every writer). Permanent, no home"),
    # --- Files that are not provenance inputs --------------------------------
    ("src/api/routes/pipeline.py", "get_ui_config", "json.load"):
        (1, _NOT_INPUT.format(what="the UI configuration")),
    ("src/api/services/training_manager.py", "TrainingManager._read_latest_metrics",
     "json.load"): (1, _NOT_INPUT.format(what="a training run's metrics, for display")),
    ("src/api/services/training_manager.py", "TrainingManager._read_progress_file",
     "json.load"): (1, _NOT_INPUT.format(what="a training run's progress, for display")),
    ("src/api/services/training_manager.py", "TrainingManager.get_metrics_history",
     "json.load"): (1, _NOT_INPUT.format(what="a training run's metrics, for display")),
    ("src/api/services/training_manager.py", "TrainingManager._test_outcome", "json.loads"):
        (1, _NOT_INPUT.format(what="a test run's own outcome file (`--outcome`), naming the "
                                   "report it published and that report's digest; kept and "
                                   "registered are read from the store and the ledger by "
                                   "`registration.report_state` (M5-1)")),
    ("src/config/runtime_presets.py", "load_runtime_settings", "json.load"):
        (1, _NOT_INPUT.format(what="a user's persisted runtime UI settings")),
    ("src/utils/gpu_lock.py", "read_holder", "json.loads"):
        (1, _NOT_INPUT.format(what="the GPU lock holder's own description, for the "
                                   "\"held by\" message (M5); the lock is the flock")),
    ("src/retrieval/backends/base.py", "VectorIndexBase.load", "json.load"):
        (1, _NOT_INPUT.format(what="a retrieval index's id map (build_index.py is outside "
                                    "the pin)")),
    ("src/webui/components/charts.py", "_versioned", "hashlib.sha256"):
        (1, _NOT_INPUT.format(what="the dashboard's own chart script, its digest in the "
                                   "script's URL so a browser never runs an old copy (M5-2)")),
    # --- The deployment probe: identification of files nobody parses there ---
    ("scripts/probe_deployment.py", "snapshot", "file_sha256"):
        (1, _PROBE + "An untouched-directory check (a location check, contract §2 rule "
            "1): before and after a refusal; records nothing as an input"),
    ("scripts/probe_deployment.py", "phase_writer._direct_control", "file_sha256"):
        (1, _PROBE + "A writer control: the digests the writer recorded against the "
            "files it wrote"),
    ("scripts/probe_deployment.py", "phase_real_build._rebuild_is_the_same_workspace",
     "file_sha256"): (2, _PROBE + "Two builds' files compared with each other (writer "
                          "determinism); neither is consumed"),
    ("scripts/probe_deployment.py", "phase_download._download", "file_sha256"):
        (1, _PROBE + "The published ontology file against the digest its loader took "
            "while parsing it"),
    # --- The deployment probe's M2 acceptance: its own outputs, read back ---
    ("scripts/probe_deployment.py", "_measurement_facts", "json.loads"):
        (1, _PROBE + "Reads back the measurement report this probe had the measurement "
            "entry point write, to check what it names; not an input"),
    ("scripts/probe_deployment.py", "_measurement_facts", "file_sha256"):
        (1, _PROBE + "The digest of that report, as evidence; nothing parses it after"),
    ("scripts/probe_deployment.py", "phase_m2_demo._inventory", "json.loads"):
        (1, _PROBE + "Reads back the inventory report this probe had the audit write"),
    ("scripts/probe_deployment.py", "phase_m2_demo._inventory", "file_sha256"):
        (1, _PROBE + "The digest of that report, as evidence"),
    ("scripts/probe_deployment.py", "phase_m2_real._inventory", "json.loads"):
        (2, _PROBE + "Reads back the two inventory reports this probe had the audit write"),
    ("scripts/probe_deployment.py", "phase_m2_real._inventory", "file_sha256"):
        (2, _PROBE + "The digests of those two reports, as evidence"),
    ("scripts/probe_deployment.py", "_training_settings", "read_once"):
        (1, _PROBE + "Phase R's training settings, parsed from the bytes of one read whose "
            "digest the report gives"),
}

_MODULES = ("hashlib", "json", "torch")
#: Primitives named by themselves wherever they come from: the hashing helper for
#: a file nobody parses, M2.1's read primitive, and the old graph loader.
_NAMED = ("file_sha256", "read_once", "load_json")


def _scope():
    for path in sorted((REPO / "src").rglob("*.py")):
        yield path
    for name in (*ENTRY_POINTS, *MOVED_IN_S9, *M4_SCRIPTS):
        yield REPO / name


def _dotted(node):
    """`a.b.c` for a chain of attributes on a name, else `None`."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _primitive(name: str):
    """The primitive `name` refers to, or `None`: anything in `hashlib`,
    `json.load(s)`, `torch.load` (also reached as `torch.serialization.load`), or a
    name in `_NAMED` however it is reached."""
    parts = name.split(".")
    if parts[-1] in _NAMED:
        return parts[-1]
    if parts[0] == "hashlib" and len(parts) > 1:
        return name
    if parts[0] in ("json", "torch") and len(parts) > 1 and parts[-1] in ("load", "loads"):
        return f"{parts[0]}.{parts[-1]}"
    return None


class _Sites(ast.NodeVisitor):
    """Every reference to a reading or hashing primitive, by enclosing function.

    **References, not only calls.** A primitive passed as a value
    (`map(file_sha256, paths)`, a default argument, `digest = hashlib.sha256`) or
    re-exported under another name (`file_sha256 = _file_sha256`) is a site, as a
    call is. **Aliases are collected before anything is counted,** from every
    import in the module wherever it sits and from assignments of a primitive to a
    name, so neither import order nor renaming hides one.
    """

    def __init__(self) -> None:
        self.stack: list = []
        self.alias: dict = {}
        self.sites: Counter = Counter()
        self.removed: list = []

    def collect(self, tree) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in _MODULES:
                        self.alias[alias.asname or root] = alias.asname and alias.name or root
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if module.split(".")[0] in _MODULES:
                        self.alias[alias.asname or alias.name] = f"{module}.{alias.name}"
                    elif alias.name in _NAMED:
                        self.alias[alias.asname or alias.name] = alias.name
        # A primitive assigned to a name makes that name an alias too; twice, so an
        # alias of an alias is followed.
        for _ in range(2):
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    resolved = self._resolve(node.value)
                    if resolved:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.alias[target.id] = resolved

    def _resolve(self, node):
        name = _dotted(node)
        if name is None:
            return None
        head, _, rest = name.partition(".")
        full = self.alias.get(head, head) + (f".{rest}" if rest else "")
        if isinstance(node, ast.Name) and head not in self.alias:
            return None
        return _primitive(full)

    def _scoped(self, node):
        if node.name in REMOVED:
            self.removed.append((node.name, node.lineno))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._scoped(node)

    def visit_AsyncFunctionDef(self, node):
        self._scoped(node)

    def visit_ClassDef(self, node):
        self._scoped(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name in REMOVED:
                self.removed.append((alias.name, node.lineno))

    def _count(self, node) -> bool:
        primitive = self._resolve(node)
        if primitive:
            self.sites[(".".join(self.stack) or "<module>", primitive)] += 1
        return bool(primitive)

    def visit_Name(self, node):
        if node.id in REMOVED:
            self.removed.append((node.id, node.lineno))
        if isinstance(node.ctx, ast.Load):
            self._count(node)

    def visit_Attribute(self, node):
        if node.attr in REMOVED:
            self.removed.append((node.attr, node.lineno))
        # The whole chain is one reference: counted once, not again at its base.
        if isinstance(node.ctx, ast.Load) and self._count(node):
            return
        self.generic_visit(node)


def _sites_of(source: str, filename: str = "<string>") -> "_Sites":
    tree = ast.parse(source, filename=filename)
    visitor = _Sites()
    visitor.collect(tree)
    visitor.visit(tree)
    return visitor


def _scan():
    sites, removed = Counter(), []
    for path in _scope():
        visitor = _sites_of(path.read_text(encoding="utf-8"), str(path))
        name = str(path.relative_to(REPO))
        for (where, primitive), count in visitor.sites.items():
            sites[(name, where, primitive)] += count
        removed += [(name, symbol, line) for symbol, line in visitor.removed]
    return sites, removed


def test_the_path_forms_are_gone_and_named_nowhere_in_scope():
    import src.evaluation.cohort as cohort
    import src.kg.artifacts as artifacts
    import src.utils.fingerprint as fingerprint

    for module in (artifacts, cohort, fingerprint):
        for name in REMOVED:
            assert not hasattr(module, name), f"{module.__name__} still defines {name}"
    _, removed = _scan()
    assert removed == [], f"a removed path form is still named: {removed}"


def test_every_path_hash_and_load_in_scope_is_a_reader_or_a_named_exception():
    sites, _ = _scan()

    unlisted = {key: n for key, n in sites.items() if key not in ALLOWED}
    assert not unlisted, (
        f"a path hash, torch.load or JSON load outside the readers and the allowlist: "
        f"{sorted(unlisted)}. Read the input through its M2.1 reader, or add the site "
        "to ALLOWED with its use and owner in review."
    )
    miscounted = {key: (sites.get(key, 0), count) for key, (count, _) in ALLOWED.items()
                  if sites.get(key, 0) != count}
    assert not miscounted, (
        f"ALLOWED no longer matches the code (found, listed): {miscounted}. Remove a "
        "stale entry, or review the new call beside a listed one."
    )


def test_every_exception_states_its_reason():
    for key, (count, reason) in ALLOWED.items():
        assert count >= 1 and len(reason) > 20, key


def test_the_scanner_sees_the_forms_it_claims_to_see():
    """Its negative controls: every form below refers to a primitive and is
    counted, so a path hash renamed, re-exported, passed as a value or imported
    after its use does not walk past the pin."""
    source = (
        "def early(p):\n"
        "    return late_alias(p)\n"
        "import json as j\n"
        "import hashlib\n"
        "import torch.serialization\n"
        "from hashlib import file_digest as fd\n"
        "from src.utils.fingerprint import file_sha256 as h\n"
        "from src.utils.fingerprint import read_once\n"
        "from src.utils import fingerprint\n"
        "file_sha256 = h\n"
        "late_alias = file_sha256\n"
        "def f(p, digest=hashlib.sha256):\n"
        "    j.loads(p.read_text()); fd(open(p, 'rb'), 'sha256'); h(p)\n"
        "    fingerprint.file_sha256(p); list(map(file_sha256, [p])); read_once(p)\n"
        "    import torch\n"
        "    torch.load(p); torch.serialization.load(p); KnowledgeGraph.load_json(p)\n"
        "    json.dumps({}); torch.save({}, p)\n"
        "def verify_graph_artifacts(d):\n"
        "    pass\n"
    )
    visitor = _sites_of(source)
    assert visitor.sites == Counter({
        ("early", "file_sha256"): 1,
        ("<module>", "file_sha256"): 2,  # the two re-exports
        ("f", "hashlib.sha256"): 1,  # the default argument
        ("f", "json.loads"): 1, ("f", "hashlib.file_digest"): 1, ("f", "file_sha256"): 3,
        ("f", "read_once"): 1, ("f", "torch.load"): 2, ("f", "load_json"): 1,
    })
    assert [name for name, _ in visitor.removed] == ["verify_graph_artifacts"]


def test_the_poison_reaches_every_path_hash_and_spares_the_named_files(monkeypatch, tmp_path):
    """The dynamic pins rest on this: under `poison_path_forms` a path hash
    returns the poison digest, however `file_sha256` was imported, and a file
    named in `true_digest_for` keeps its real digest. If the fixture stopped
    reaching `file_sha256`, every dynamic pin would pass without poisoning
    anything."""
    import hashlib

    import src.utils.fingerprint as fingerprint
    from src.utils.fingerprint import file_sha256 as bound_at_import
    from tests.fixtures.path_forms import POISON, poison_path_forms

    poisoned, spared = tmp_path / "poisoned.json", tmp_path / "kg.json"
    poisoned.write_text("{}")
    spared.write_text("[]")
    poison_path_forms(monkeypatch, true_digest_for=[spared])

    assert fingerprint.file_sha256(poisoned) == POISON
    assert bound_at_import(poisoned) == POISON
    assert fingerprint.file_sha256(spared) == hashlib.sha256(b"[]").hexdigest()
    # The read primitive is untouched: it hashes the bytes it returns.
    assert fingerprint.read_once(poisoned).sha256 == hashlib.sha256(b"{}").hexdigest()


def test_nothing_in_scope_binds_the_hook_the_poison_sits_on():
    """The poison replaces `hashlib.file_digest` on the module. A name bound by
    `from hashlib import file_digest` keeps the real function, so a path hashed
    through it would never see the poison; nothing in scope may bind it."""
    for path in _scope():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "hashlib":
                names = {alias.name for alias in node.names}
                assert "file_digest" not in names and "*" not in names, (
                    f"{path.relative_to(REPO)}:{node.lineno} binds hashlib.file_digest"
                )


def test_the_entry_points_import_no_hashing_primitive_at_module_level():
    """The entry points depend on no hashing primitive at module level: what
    they record comes from the readers they call."""
    for name in ENTRY_POINTS:
        tree = ast.parse((REPO / name).read_text(encoding="utf-8"))
        bound = {alias.asname or alias.name for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        assert not bound & {"file_sha256", "hashlib", "read_once"}, (name, bound)
