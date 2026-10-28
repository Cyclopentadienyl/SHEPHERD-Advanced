"""
README and the training playbook work as written (BACKLOG 45).
==============================================================
`README.md` and `docs/TRAINING_PIPELINE_PLAYBOOK.md` are the operator's entry
documents. They used to start a bare `launch_shepherd.sh` (read-only, no
dataset listed, every job refused), send the operator to the retired "Training
Console" and its "Data Dir", and give `measure_scorer.py` without the arguments
it requires. What these pin, through the code itself:

- **every command they give parses** with its script's own argument parser,
  the script's checks after argparse's included (`measure_scorer.parse_args`);
- **every page and field they name is the dashboard's**: the navigation of the
  built app (`create_gradio_app`), the research groups the shell states, the
  Training page's labels and buttons; the retired names appear nowhere;
- **a section that gives a command needing the service's settings points to
  the deployment guide for them**, naming the variable: the launcher (§4.12),
  a script that takes the GPU lock (`take_gpu_lock` in its source; §4.11, the
  workspaces root), a registration or a `--cohort` measurement (§4.11, the
  registry). The rules themselves stay in the guide, in one place, and the
  guide's sections cited hold them;
- **the playbook's safety net is the load's refusal** (contract M2.4): the
  refusals it quotes are what serving raises for a checkpoint of another graph
  and for one with no record, and the structural fingerprint is the drawer's
  diagnostic beside them, not the safety net it used to be called.

Module: tests/unit/test_entry_documents.py
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from src.config import server_settings as ss

REPO = Path(__file__).resolve().parents[2]
DOCUMENTS = ("README.md", "docs/TRAINING_PIPELINE_PLAYBOOK.md")

_SCRIPT = re.compile(r"(?:^|/)scripts/(\w+)\.py$")
_LAUNCHER = re.compile(r"(?:^|/)launch_shepherd\.(?:sh|cmd)$")
_SPAN = re.compile(r"`([^`\n]+)`")


def _text(document: str) -> str:
    return (REPO / document).read_text(encoding="utf-8")


def _sections(text: str) -> List[Tuple[str, str]]:
    """(heading, body) for each `##` section, its subsections included; the
    text before the first is the section ''."""
    parts = re.split(r"^(## .*)$", text, flags=re.MULTILINE)
    out = [("", parts[0])]
    out += [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    return out


def _tokens(line: str) -> List[str]:
    """A command line's words as a shell splits them, without its comment;
    `[...]`, the documents' optional part, is given as written inside."""
    try:
        return shlex.split(line.replace("[", " ").replace("]", " "), comments=True)
    except ValueError:  # an unbalanced quote: not a command line
        return []


#: A fenced block: its language and its text. A shell block is a command
#: block; an untagged one (the pipeline diagrams) is not.
_FENCE = re.compile(r"^[ \t]*```(\w*)[^\n]*\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL)
_SHELL = {"bash", "sh", "shell", "console", "cmd", "bat"}


def _command_lines(body: str) -> List[Tuple[List[str], bool]]:
    """The command lines of a section, each with whether it is in a fenced
    block: each line of a shell block (its `\\` continuations joined), and
    each inline code span outside the blocks."""
    lines: List[Tuple[List[str], bool]] = []
    for language, block in _FENCE.findall(body):
        if language in _SHELL:
            lines += [(_tokens(line), True)
                      for line in block.replace("\\\n", " ").splitlines()]
    outside = _FENCE.sub("", body)
    lines += [(_tokens(span), False) for span in _SPAN.findall(outside)]
    return [(line, block) for line, block in lines if line]


def _script_commands(body: str) -> List[Tuple[str, List[str]]]:
    """(script name, its arguments) for each command line naming a script: a
    line of a fenced block, or an inline span that gives the script something
    (a bare `scripts/x.py` in the text is a reference, not a command)."""
    found = []
    for line, block in _command_lines(body):
        for i, token in enumerate(line):
            match = _SCRIPT.search(token)
            if match:
                if block or line[i + 1:]:
                    found.append((match.group(1), line[i + 1:]))
                break
    return found


def _launches(body: str) -> bool:
    return any(_LAUNCHER.search(token) for line, _block in _command_lines(body) for token in line)


def _all_commands() -> List[Tuple[str, str, List[str]]]:
    return [(document, script, argv)
            for document in DOCUMENTS
            for _heading, body in _sections(_text(document))
            for script, argv in _script_commands(body)]


class _Parsed(Exception):
    """Raised by argparse's own parse, once it has accepted the arguments, to
    stop a script whose parser is built inside its `main`."""


def _parse(script: str, argv: List[str], monkeypatch) -> object:
    """The script's own parse of `argv`: its `parse_args` (with `argv`, or
    from `sys.argv` when it reads that), else its `main` stopped right after
    argparse accepted the arguments. argparse's refusal is `SystemExit(2)`."""
    source = (REPO / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    if "ArgumentParser" not in source:
        assert argv == [], f"scripts/{script}.py takes no arguments, given {argv}"
        return None
    module = importlib.import_module(f"scripts.{script}")
    monkeypatch.setattr(sys, "argv", [f"scripts/{script}.py", *argv])
    entry = getattr(module, "parse_args", None)
    if entry is None:
        original = argparse.ArgumentParser.parse_args

        def parse_then_stop(self, args=None, namespace=None):
            raise _Parsed(original(self, args, namespace))

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
        entry = module.main
    try:
        return entry(argv) if inspect.signature(entry).parameters else entry()
    except _Parsed as parsed:
        return parsed.args[0]


COMMANDS = _all_commands()


def test_the_documents_give_the_commands_they_are_checked_for():
    """The commands BACKLOG 45 named are among those parsed below, each with
    the arguments its parser requires."""
    given = {(script, argv[0] if argv else "") for _doc, script, argv in COMMANDS}
    for script, first in (("train_model", "--data-dir"), ("measure_scorer", "--checkpoint"),
                          ("import_cohort", "--registry"), ("record_evaluation", "--report")):
        assert (script, first) in given, (script, first, sorted(given))


@pytest.mark.parametrize("document, script, argv", COMMANDS,
                         ids=[f"{d}:{s}:{' '.join(a)}" for d, s, a in COMMANDS])
def test_every_command_parses_with_its_script_s_own_parser(document, script, argv, monkeypatch):
    try:
        _parse(script, argv, monkeypatch)
    except SystemExit as exc:
        pytest.fail(f"{document}: scripts/{script}.py {' '.join(argv)} is refused by its "
                    f"parser (exit {exc.code})")


# ---------------------------------------------------------------------------
# The pages and fields they name
# ---------------------------------------------------------------------------
#: The Training and Fine-tune pages' fields and actions the playbook names, as
#: `**name**` (K6: the base form's labels; the ShepherdGNN form's are gone).
TRAINING_NAMES = ("Dataset", "Run name", "Epochs", "Learning rate", "Batch size", "Seed",
                  "Continue…", "Check checkpoint", "Continue Training", "Stop", "Tasks")
#: Names of the retired ShepherdGNN form the documents no longer teach (K6).
RETIRED_FORM = ("GNN Conv Type", "Max Subgraph Nodes", "Batch Size")
#: Names of what the dashboard no longer has (M5-2).
RETIRED = ("training console", "data dir", "diagnosis panel")


@pytest.fixture(scope="module")
def shown():
    """Every label and value the built dashboard shows (its tabs, fields and
    buttons)."""
    pytest.importorskip("gradio")
    from src.webui.app import create_gradio_app

    texts = []
    for block in create_gradio_app().blocks.values():
        for attribute in ("label", "value"):
            value = getattr(block, attribute, None)
            if isinstance(value, str):
                texts.append(value.strip())
    return texts


def _research_groups() -> Dict[str, List[str]]:
    from src.webui import shell

    return {group: [page.strip() for page in pages.split(",")]
            for group, pages in re.findall(r"(\w+) \(([^)]*)\)", shell.RESEARCH_TABLIST_LABEL)}


@pytest.mark.parametrize("document", DOCUMENTS)
def test_no_retired_page_or_field_is_named(document):
    text = _text(document).lower()
    assert [name for name in RETIRED if name in text] == []


@pytest.mark.parametrize("document", DOCUMENTS)
def test_every_research_page_named_is_the_dashboard_s(document, shown):
    groups = _research_groups()
    plain = _text(document).replace("**", "")
    # A page name may hold a hyphen ("Fine-tune").
    named = re.findall(r"Research(?: \(pre-clinical\))? › (\w+) › ([\w-]+)", plain)
    assert named, f"{document} names no research page"
    for group, page in named:
        assert page in groups.get(group, []), (document, group, page, groups)
        assert page in shown, page
    for page in re.findall(r"\*\*(Research \(pre-clinical\)|Diagnosis \(clinical\))\*\*", _text(document)):
        assert page in shown, page


def test_the_training_page_s_names_in_the_playbook_are_its_labels_and_buttons(shown):
    playbook = _text("docs/TRAINING_PIPELINE_PLAYBOOK.md")
    for name in TRAINING_NAMES:
        assert re.search(rf"\*\*{re.escape(name)}(?: = [^*]*)?\*\*", playbook), name
        assert any(text == name or text.endswith(f" {name}") for text in shown), name
    for name in RETIRED_FORM:
        assert f"**{name}" not in playbook, name
        assert name not in shown, name


# ---------------------------------------------------------------------------
# The settings: pointed to, not copied
# ---------------------------------------------------------------------------
def _takes_the_gpu_lock(script: str) -> bool:
    path = REPO / "scripts" / f"{script}.py"
    return path.exists() and "take_gpu_lock(" in path.read_text(encoding="utf-8")


def _needs_the_registry(script: str, argv: List[str]) -> bool:
    return (script == "record_evaluation" and "--report" in argv) or (
        script == "measure_scorer" and ("--cohort" in argv or "--record" in argv))


def _needs(body: str) -> Dict[str, set]:
    """The variables a section's commands need, each with the guide section
    stating it."""
    needs: Dict[str, set] = {}
    if _launches(body):
        needs.setdefault("§4.12", set()).update({ss.ENV_DEPLOYMENT_MODE, ss.ENV_WORKSPACES_ROOT})
    for script, argv in _script_commands(body):
        if _takes_the_gpu_lock(script):
            needs.setdefault("§4.11", set()).add(ss.ENV_WORKSPACES_ROOT)
        if _needs_the_registry(script, argv):
            needs.setdefault("§4.11", set()).add(ss.ENV_REGISTRY)
    return needs


@pytest.mark.parametrize("document", DOCUMENTS)
def test_a_command_that_needs_the_service_s_settings_points_to_the_guide(document):
    checked = 0
    for heading, body in _sections(_text(document)):
        for section, variables in _needs(body).items():
            checked += 1
            assert section in body, f"{document} {heading!r} does not point to the guide's {section}"
            missing = sorted(v for v in variables if v not in body)
            assert missing == [], f"{document} {heading!r} does not name {missing}"
    assert checked, f"{document}: no command needing the settings was found"


@pytest.mark.parametrize("guide", ["deployment-guide.en.md", "deployment-guide.md"])
def test_the_guide_s_sections_cited_state_those_settings(guide):
    """The pointers lead to the rules: §4.11 states the workspaces root and the
    registry for the shell, §4.12 the service's settings."""
    sections = {heading.split()[1]: body for heading, body in _sections(_text(guide))
                if re.match(r"## 4\.1[12] ", heading)}
    assert set(sections) == {"4.11", "4.12"}, sorted(sections)
    assert ss.ENV_WORKSPACES_ROOT in sections["4.11"] and ss.ENV_REGISTRY in sections["4.11"]
    for variable in (ss.ENV_DEPLOYMENT_MODE, ss.ENV_WORKSPACES_ROOT, ss.ENV_REGISTRY):
        assert variable in sections["4.12"], variable


# ---------------------------------------------------------------------------
# The safety net: what a load does with a checkpoint of another graph
# ---------------------------------------------------------------------------
def _safety_net() -> str:
    """The playbook's safety-net section, its lines joined."""
    body = next((body for heading, body in _sections(_text("docs/TRAINING_PIPELINE_PLAYBOOK.md"))
                 if heading.startswith("## The safety net")), None)
    assert body is not None, "the playbook has no section '## The safety net…'"
    return " ".join(body.split())


def test_the_playbook_s_safety_net_is_the_load_s_refusal_in_the_code_s_words(tmp_path):
    """Since contract M2.4 a load refuses, before any model is built, a
    checkpoint whose record names another graph or none; the structural
    fingerprint is a diagnostic beside that refusal. The playbook said the
    fingerprint's warnings were the safety net and that old checkpoints "will
    warn" after a rebuild. What it quotes now is what the real load raises for
    both cases, and what the Diagnosis page's drawer shows for the fingerprint."""
    pytest.importorskip("torch")
    from src.api import main as service
    from src.inference.pipeline import PipelineBuildError
    from src.kg.artifacts import GRAPH_ARTIFACTS
    from src.utils.fingerprint import verify_fingerprint
    from src.webui.components import diagnosis_card, model_list
    from tests.unit import test_training_record as record

    text = _safety_net()
    data_dir, checkpoint = record.build_workspace(tmp_path / "a")
    refused = {}
    for case, workspace, served in (
            ("another graph", record._same_shaped_other(tmp_path / "b"), checkpoint),
            ("no record", data_dir, record._pre_contract(checkpoint, tmp_path / "old.pt"))):
        with pytest.raises(PipelineBuildError) as refusal:
            record._serve(workspace, served)
        refused[case] = str(refusal.value)
    for case, quoted in (("another graph", "does not match this workspace's graph"),
                         ("another graph", "node_features differs"),
                         ("no record", "records no training_input_digests")):
        assert quoted in refused[case], (case, refused[case])
        assert quoted in text, quoted
    assert "refused before any model is built" in text
    assert all(f"`{name}`" in text for name in GRAPH_ARTIFACTS.values()), GRAPH_ARTIFACTS
    assert "Startup failed" in inspect.getsource(service.lifespan)
    assert "Not reloaded; the model in use is unchanged." in text
    assert "Not reloaded; the model in use is unchanged." in inspect.getsource(model_list)

    # The fingerprint: never a refusal; its two lines as the drawer writes them.
    differs = verify_fingerprint({"data_fingerprint": {"node_types": ["gone"]}}, {})
    legacy = verify_fingerprint({}, {})
    for warnings, line in ((differs, "Graph structure differs from the checkpoint's structural "
                                     "fingerprint"),
                           (legacy, "Graph structure not compared")):
        assert line in diagnosis_card._structure_line({"fingerprint_warnings": warnings}), line
        assert f'"{line}"' in text, line
    assert "never stops a load" in text

    # Troubleshooting sends a rebuild's old checkpoint to that refusal, not to
    # the fingerprint's warnings.
    rows = [row for row in _text("docs/TRAINING_PIPELINE_PLAYBOOK.md").splitlines()
            if row.startswith("|") and "rebuild" in row.split("|")[1]]
    assert rows and all("does not match this workspace's graph" in row for row in rows), rows
