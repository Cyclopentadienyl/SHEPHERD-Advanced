"""
The application mounts the dashboard whatever was imported first.
=================================================================
BACKLOG 35 (M5-3 review EVAL-5): `src/api/__init__.py` built the application on
import, so importing a dashboard page first (it imports the API's services)
started building the app while `src.webui.app` was half imported; the mount
then failed as a circular import, logged as "Gradio not available", and the
app had no ``/ui`` (the probe's gated-event test failed when collected after a
page's tests). The package now builds nothing on import. Each order runs in a
fresh interpreter.

Module: tests/unit/test_api_import_order.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("gradio")

REPO_ROOT = Path(__file__).resolve().parents[2]

PROBE = r'''
import importlib, json, logging, sys
records = []
class Keep(logging.Handler):
    def emit(self, record):
        records.append((record.levelname, record.getMessage()))
logging.getLogger().addHandler(Keep())
logging.getLogger().setLevel(logging.INFO)
for name in sys.argv[1:]:
    importlib.import_module(name)
import src.api.main as main
from src.api import app
mounted = [getattr(r, "path", "") for r in main.app.routes if getattr(r, "path", "") == "/ui"]
print("RESULT" + json.dumps({"mounted": mounted, "same_app": app is main.app,
                             "problems": [m for level, m in records
                                          if "dashboard" in m and level in ("WARNING", "ERROR")]}))
'''


@pytest.mark.parametrize("first", [
    ["src.webui.components.test_page"],
    ["src.webui.app"],
    ["src.api.services.training_manager"],
    ["src.webui.components.models_page", "src.webui.components.training_console"],
    [],
], ids=["test-page", "webui-app", "a-service", "two-pages", "the-service-order"])
def test_the_dashboard_is_mounted_whatever_was_imported_first(first):
    done = subprocess.run([sys.executable, "-c", PROBE, *first], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, timeout=600,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    line = next((ln for ln in done.stdout.splitlines() if ln.startswith("RESULT")), None)
    assert line is not None, done.stdout[-2000:] + done.stderr[-4000:]
    result = json.loads(line[len("RESULT"):])
    assert result == {"mounted": ["/ui"], "same_app": True, "problems": []}, result


BLOCKED = r'''
import importlib.abc, json, logging, sys
records = []
class Keep(logging.Handler):
    def emit(self, record):
        records.append((record.levelname, record.getMessage()))
logging.getLogger().addHandler(Keep())
logging.getLogger().setLevel(logging.INFO)
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "src.webui.app":
            raise ImportError("cannot import name 'x' from partially initialized module")
blocked = sys.argv[1]
if blocked == "no-gradio":
    sys.modules["gradio"] = None
elif blocked == "no-page":
    sys.modules["src.webui.app"] = None
else:
    sys.meta_path.insert(0, Refuse())
import src.api.main as main
mounted = [getattr(r, "path", "") for r in main.app.routes if getattr(r, "path", "") == "/ui"]
print("RESULT" + json.dumps({"mounted": mounted,
                             "said": [level for level, m in records if "dashboard" in m
                                      and level in ("WARNING", "ERROR")]}))
'''


@pytest.mark.parametrize("blocked, said", [
    ("no-gradio", ["WARNING"]),
    ("no-page", ["ERROR"]),
    ("import-error", ["ERROR"]),
], ids=["gradio-not-installed", "page-module-missing", "page-import-fails"])
def test_a_dashboard_that_cannot_be_imported_is_logged_as_a_defect_unless_gradio_is_absent(
        blocked, said):
    """A missing Gradio is a deployment choice (a warning); any other import
    failure of the dashboard is a defect and is logged as an error (EVAL-5)."""
    done = subprocess.run([sys.executable, "-c", BLOCKED, blocked], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, timeout=600,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    line = next((ln for ln in done.stdout.splitlines() if ln.startswith("RESULT")), None)
    assert line is not None, done.stdout[-2000:] + done.stderr[-4000:]
    assert json.loads(line[len("RESULT"):]) == {"mounted": [], "said": said}


def test_importing_the_package_builds_no_application():
    code = ("import sys, src.api; "
            "sys.exit(1 if 'src.api.main' in sys.modules else 0)")
    done = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, timeout=600)
    assert done.returncode == 0
