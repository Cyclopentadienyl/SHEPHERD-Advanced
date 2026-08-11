"""
The vector index stays detached from diagnosis.
===============================================
The subsystem under `src/retrieval/` is implemented, tested, and deliberately
kept — but by institutional decision nothing on the diagnosis path may depend on
it. `.import-linter.ini` enforces that at the import level.

Imports are not the only way back in. A field could reappear on `PipelineConfig`,
a key in the status payload, a parameter on the factory, or an environment read
in the service — none of which needs a new import of `src.retrieval` to
reconnect the subsystem. This file pins those surfaces, so the two checks
together cover reconnection by import *and* by configuration. Neither subsumes
the other.

The five removed names are pinned exactly, on the exact surfaces they were
removed from. The string `SHEPHERD_VECTOR_INDEX_PATH` is checked only in
executable code: the migration note and the findings document must be free to
mention it, and a global ban would make correcting the record impossible.

AST rather than imports: this runs in any environment, needs no torch or PyG,
and takes milliseconds. A runtime cross-check runs additionally wherever the
modules happen to be importable, since AST alone cannot see a field arriving
through inheritance.
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = REPO_ROOT / "src" / "inference" / "pipeline.py"
ROUTES = REPO_ROOT / "src" / "api" / "routes" / "pipeline.py"
SRC = REPO_ROOT / "src"

CONFIG_FIELDS = ("vector_index_path", "ann_top_k", "ann_score_threshold")
STATUS_KEYS = ("vector_index_ready", "vector_index_size")
ENV_VAR = "SHEPHERD_VECTOR_INDEX_PATH"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found — has it been renamed?")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found — has it been renamed?")


def _annotated_field_names(cls: ast.ClassDef) -> set:
    return {
        stmt.target.id
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }


@pytest.mark.parametrize("field", CONFIG_FIELDS)
def test_pipeline_config_has_no_vector_index_field(field):
    """PipelineConfig must not offer a way to switch the subsystem back on."""
    fields = _annotated_field_names(_class(_tree(PIPELINE), "PipelineConfig"))
    assert field not in fields, (
        f"PipelineConfig.{field} is back. The vector index is detached from diagnosis "
        f"by decision; reconnecting it is a reviewed change, not a config addition."
    )


@pytest.mark.parametrize("key", STATUS_KEYS)
def test_pipeline_status_payload_has_no_vector_index_key(key):
    """get_pipeline_config() builds the status payload from a dict literal."""
    fn = _function(_tree(PIPELINE), "get_pipeline_config")
    keys = {
        node.value
        for ret in ast.walk(fn)
        if isinstance(ret, ast.Dict)
        for node in ret.keys
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert key not in keys, f"get_pipeline_config() reports {key!r} again"


@pytest.mark.parametrize("key", STATUS_KEYS)
def test_status_response_schema_has_no_vector_index_field(key):
    """The public /pipeline/status schema is the surface an external client sees."""
    fields = _annotated_field_names(_class(_tree(ROUTES), "PipelineStatusResponse"))
    assert key not in fields, f"PipelineStatusResponse.{key} is back in the public schema"


def test_pipeline_factory_takes_no_vector_index_path():
    """create_diagnosis_pipeline must not accept the path again."""
    fn = _function(_tree(PIPELINE), "create_diagnosis_pipeline")
    args = fn.args
    names = {
        a.arg
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]
    }
    assert "vector_index_path" not in names, (
        "create_diagnosis_pipeline accepts vector_index_path again"
    )


def test_no_executable_module_reads_the_env_var():
    """Checked in code only.

    The findings document and any migration note must stay free to name the
    variable — a global ban on the string would make it impossible to record what
    was removed.
    """
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if ENV_VAR in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, (
        f"{ENV_VAR} is read again by: {offenders}. Diagnosis start-up must not be able "
        f"to load a vector index from the environment."
    )


def test_runtime_objects_agree_with_the_source():
    """Cross-check the live objects where the environment can import them.

    AST cannot see a field arriving through inheritance or assigned dynamically;
    this catches that. It is additional to the checks above, not a replacement —
    it skips wherever torch/PyG are absent, and the pins must hold there too.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import inspect

    from src.inference.pipeline import PipelineConfig, create_diagnosis_pipeline

    assert not (set(PipelineConfig.__dataclass_fields__) & set(CONFIG_FIELDS))
    assert "vector_index_path" not in inspect.signature(create_diagnosis_pipeline).parameters
