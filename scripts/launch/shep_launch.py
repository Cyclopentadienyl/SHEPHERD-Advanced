# scripts/launch/shep_launch.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, errno, html, json, os, platform, socket, subprocess, sys, textwrap, threading, time, webbrowser
from pathlib import Path
from typing import Callable, Dict, Any, List, Mapping, NamedTuple, Optional, Tuple
import shlex
from urllib.request import ProxyHandler, build_opener
from urllib.error import URLError

REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/launch -> scripts -> REPO_ROOT
CONFIG_DIR = REPO_ROOT / "configs"  # aligned with v3 structure
ACCEL_TABLE = CONFIG_DIR / "accelerators.json"
DEPLOYMENT_CONFIG = CONFIG_DIR / "deployment.yaml"
DEFAULT_ENTRY = "uvicorn"
UVICORN_APP = "src.api.main:app"
# Loopback by default (SPEC_4 B1, since M5-1); another --host, --fd or --uds
# after `--` is checked by the bind guard below (`bind_plan`).
UVICORN_DEFAULT_ARGS = ["--host", "127.0.0.1", "--port", "8000"]
#: The port a start without ``--port`` takes, and the ones it tries next, in
#: order, when that one is in use (plan §4.5; `choose_port`).
DEFAULT_PORT = 8000
FALLBACK_PORTS = range(8001, 8020)
# Always last on uvicorn's command line, so a --proxy-headers passed through
# cannot turn it back on: uvicorn would otherwise rewrite the client address
# from X-Forwarded-For for a peer on 127.0.0.1. M1 has no proxy, and D8 reads
# the socket's own address and the headers, never the client address.
UVICORN_FORCED_ARGS = ["--no-proxy-headers"]

# Ensure repo root is importable so the shared, gradio-free runtime presets
# module loads regardless of how the launcher is invoked.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.config.runtime_presets import (  # noqa: E402
    ALLOC_SOURCE_ENV,
    allocator_env,
    load_runtime_settings,
    resolve_allocator,
)
from src.config.atomic_json import ensure_directory, write_text_atomic  # noqa: E402
from src.config.deployment_settings import ENV_SETTINGS_FILE, settings_location  # noqa: E402
from src.config.server_settings import (  # noqa: E402
    ENV_BIND_HOST,
    ENV_DEPLOYMENT_MODE,
    ENV_LOCAL_ENTRY,
    LOCAL_ENTRY,
    MODE_OFF_STATE,
    bind_is_loopback,
    deployment_mode,
    inherited_socket_bind,
    mode_off_why,
    network_bind,
)
from src.utils import launch_instance as instances  # noqa: E402

#: The launcher's own variables, never an operator's: each is removed from its
#: environment before anything reads it and set again only by this start
#: (the bind host, the local start's mark, the instance's id and info file).
LAUNCHER_VARIABLES = (ENV_BIND_HOST, ENV_LOCAL_ENTRY, instances.ENV_INSTANCE_ID,
                      instances.ENV_INSTANCE_FILE)
#: The script a desktop entry starts, and the icon it shows.
LAUNCH_SCRIPT = REPO_ROOT / "launch_shepherd.sh"
DESKTOP_ICON = REPO_ROOT / "scripts" / "launch" / "shepherd-advanced.svg"

def log(msg: str) -> None:
    print(f"[SHEPHERD] {msg}")

def run(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check)

def serve(cmd: List[str]) -> int:
    """Run the server beside this process until it exits; its exit status.
    Only where this process cannot become the server (`can_hand_over`:
    Windows, where ``os.exec*`` changes the PID). There Ctrl-C and Ctrl-Break
    are console events that reach the server too, since it shares this
    console, and a SIGTERM from another process is TerminateProcess, which no
    process can catch or pass on, so nothing is forwarded.

    **A Ctrl-C waits for the server, never kills it.** The server is in this
    terminal's process group and gets the same SIGINT; it then stops the GPU
    job it runs, which leads a process group of its own and so no longer gets
    the terminal's signal (`src/api/main.py`, review of 8242b47).
    `subprocess.run` would kill the server 0.25 s after the first Ctrl-C,
    before that stop ends, and leave the job running. A second Ctrl-C makes
    uvicorn stop waiting for open connections; the server still waits for the
    job's stop before it exits.
    """
    log("$ " + " ".join(cmd))
    process = subprocess.Popen(cmd)
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            continue

# No proxy, the environment's or the Windows system one: an operator's
# http_proxy (the deployment guide sets one for downloads) would take the
# loopback /health poll, and no_proxy=localhost does not cover 127.0.0.1.
_LOCAL_OPENER = build_opener(ProxyHandler({}))

def urlopen(url: str, timeout: float):
    """Open ``url`` on this host's own server, never through a proxy."""
    return _LOCAL_OPENER.open(url, timeout=timeout)

def _answers_as(response: Any, instance_id: Optional[str]) -> bool:
    """Whether a 200 from ``/health`` is this start's server: with no
    ``instance_id`` to compare, any 200 is; with one, only an answer that
    names it (another program may hold the port; plan §4.5)."""
    if response.status != 200:
        return False
    if instance_id is None:
        return True
    try:
        body = json.loads(response.read().decode("utf-8"))
    except (ValueError, OSError, AttributeError):
        return False
    return isinstance(body, dict) and body.get("instance_id") == instance_id


#: What a start says once it asked this computer's browser for a page
#: (`open_in_browser`): a browser was started for it, or none could be.
BROWSER_OPENED = "Opened {url} in the browser."
NO_BROWSER = "No browser could be opened on this computer: open {url} in a browser."


def open_in_browser(url: str) -> bool:
    """**Open ``url`` in this computer's browser and say what happened**: the
    one open of every start (the first start's once its server answers, a
    second start's of the instance that already runs, a failed start's error
    page). ``webbrowser.open`` returns False when it finds no browser (a host
    with no desktop session, an SSH login) and can raise; then the log says to
    open the address instead (T08: a second start printed "Opening it." while
    nothing opened). True means a browser program was started for it, which
    is all ``webbrowser`` can tell; it is never raised from."""
    try:
        opened = bool(webbrowser.open(url))
    except Exception:  # noqa: BLE001 - any failure to start a browser is "not opened"
        opened = False
    log((BROWSER_OPENED if opened else NO_BROWSER).format(url=url))
    return opened


def _open_browser_when_ready(url: str, health_url: str, progress_every: float = 30.0,
                             server: Optional[int] = None, instance_id: Optional[str] = None,
                             on_gone: Optional[Callable[[], None]] = None) -> bool:
    """Poll ``health_url`` until it answers, then open ``url`` once
    (`open_in_browser`, which says whether a browser opened it); True once
    the server answered, False when it ended first.

    Runs beside the server: in the helper `start_browser_opener` starts,
    given the server's PID as ``server``, or on Windows in a daemon thread of
    the launcher that waits for the server. **There is no deadline.**
    uvicorn runs the app's lifespan, where an eager pipeline build loads,
    before it binds the socket, so the first answer is the end of startup
    however long the load takes. A fixed 60 s bound only guessed that time,
    and gave up on a start that loads a large SP table eagerly (about 84 s
    on a GB10; BACKLOG item 17). The wait cannot outlive the server: the
    helper returns once no process ``server`` is left, and the daemon thread
    ends with the launcher, which exits when the server does. A line about
    every ``progress_every`` seconds says the server is still starting.

    ``/health``, not ``/ready``: ``/ready`` stays 503 after a failed startup,
    and the dashboard is where an operator then reloads a pipeline. With
    ``instance_id`` (the launcher's start), only an answer naming that id
    counts (`_answers_as`). ``on_gone`` runs when the server ended before it
    answered: a start without a terminal shows its error page then.
    """
    started = time.monotonic()
    next_progress = progress_every
    while server is None or _is_running(server):
        try:
            resp = urlopen(health_url, timeout=3)
            if _answers_as(resp, instance_id):
                log("Server ready.")
                open_in_browser(url)
                return True
        except (URLError, OSError):
            pass
        elapsed = time.monotonic() - started
        if elapsed >= next_progress:
            log(f"Server still starting ({elapsed:.0f} s); the browser opens when {health_url} answers")
            next_progress = elapsed + progress_every
        time.sleep(1.0)
    if on_gone is not None:
        on_gone()
    return False

def _is_running(pid: int) -> bool:
    """Whether a process ``pid`` exists (one of another user's counts)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

#: The browser opener as a program of its own (`start_browser_opener`). Its
#: first process forks and exits at once, so the launcher, which waits for
#: that one, has no child left when it becomes the server; the second, whose
#: parent is then the system's, polls (`_open_browser_when_ready`). It stays
#: in the terminal's process group, so a Ctrl-C ends it too. Its optional
#: arguments: the instance id ``/health`` must answer, and the error page and
#: log of a start without a terminal (`server_gone_page`), shown when the
#: server ends before it answers.
_BROWSER_OPENER = (
    "import importlib.util, os, sys\n"
    "if os.fork():\n"
    "    os._exit(0)\n"
    "try:\n"
    "    spec = importlib.util.spec_from_file_location('shep_launch', sys.argv[1])\n"
    "    launcher = importlib.util.module_from_spec(spec)\n"
    "    spec.loader.exec_module(launcher)\n"
    "    instance_id, page, log = (sys.argv[5:] + ['', '', ''])[:3]\n"
    "    on_gone = launcher.server_gone_page(page, log) if page else None\n"
    "    launcher._open_browser_when_ready(sys.argv[2], sys.argv[3], server=int(sys.argv[4]),\n"
    "                                      instance_id=instance_id or None, on_gone=on_gone)\n"
    "except KeyboardInterrupt:\n"
    "    pass\n"
)

def start_browser_opener(url: str, health_url: str, instance_id: Optional[str] = None,
                         failure_page: Optional[Path] = None,
                         log_path: Optional[Path] = None) -> None:
    """Start the browser opener detached from this process, which then becomes
    the server (`exec_server`) and keeps its PID, the one the opener is given
    to watch. A new interpreter rather than a fork of this one: the launcher
    has imported torch, whose threads a forked copy would not have."""
    sys.stdout.flush()
    extra = [instance_id or "", str(failure_page or ""), str(log_path or "")]
    subprocess.run([sys.executable, "-c", _BROWSER_OPENER, str(Path(__file__).resolve()),
                    url, health_url, str(os.getpid()), *extra], check=False)

def apply_allocator(env, settings: Dict[str, Any]) -> List[str]:
    """Put the CUDA allocator into ``env`` and return what to log.

    The rule is ``allocator_env``, the one Restart Backend applies, so the
    launcher and a restart cannot disagree about a saved preset. What is added
    here is only the launcher's reporting, and marking an explicit override
    ``"env"`` so the choice is visible; the rule keeps an unmarked one too.
    """
    resolved = allocator_env(env, settings)
    if resolved.get(ALLOC_SOURCE_ENV) == "preset":
        requested = settings.get("allocator_preset")
        preset, _conf = resolve_allocator(requested)
        messages = []
        if requested is not None and requested != preset:
            messages.append(
                f"WARNING: unknown allocator preset '{requested}'; falling back to '{preset}'"
            )
        env["PYTORCH_ALLOC_CONF"] = resolved["PYTORCH_ALLOC_CONF"]
        env[ALLOC_SOURCE_ENV] = "preset"
        messages.append(
            f"Runtime: PYTORCH_ALLOC_CONF={resolved['PYTORCH_ALLOC_CONF']} "
            f"(allocator preset={preset})"
        )
        return messages
    env.setdefault(ALLOC_SOURCE_ENV, "env")
    return ["Runtime: PYTORCH_ALLOC_CONF set in environment — respecting explicit override"]

def read_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def read_yaml(path: Path) -> Dict[str, Any]:
    """Read YAML config file."""
    if path.exists():
        try:
            import yaml
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            log("WARNING: PyYAML not installed, cannot read deployment.yaml")
    return {}

def get_platform_key() -> str:
    """Get platform key for deployment config lookup (e.g., linux_x86_64)."""
    os_name = "windows" if sys.platform.startswith("win") else "linux"
    arch = platform.machine().lower()
    # Normalize arch names
    if arch in {"amd64", "x86_64"}:
        arch = "x86_64"
    elif arch in {"aarch64", "arm64"}:
        arch = "aarch64"
    return f"{os_name}_{arch}"

def get_deployment_config() -> Dict[str, Any]:
    """Load deployment config with platform-specific overrides applied."""
    config = read_yaml(DEPLOYMENT_CONFIG)
    if not config:
        return {}

    defaults = config.get("defaults", {})
    platform_key = get_platform_key()
    platform_overrides = config.get("platforms", {}).get(platform_key, {})

    # Deep merge: platform overrides take precedence
    def deep_merge(base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = deep_merge(defaults, platform_overrides)
    merged["_platform"] = platform_key
    merged["_indexing"] = config.get("indexing", {})
    merged["_paths"] = config.get("paths", {})
    return merged

def pep440_python() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}"

def detect_torch() -> Tuple[Optional[str], Optional[str]]:
    try:
        import torch  # type: ignore
        cuda = torch.version.cuda or ""
        return torch.__version__, cuda
    except Exception:
        return None, None

def is_arm() -> bool:
    return platform.machine().lower() in {"aarch64", "arm64"}

def is_windows() -> bool:
    return sys.platform.startswith("win")

def export_lock_constraints(destination: Path) -> Optional[str]:
    """Write the project's locked set to `destination` as pip constraints,
    exported from uv.lock by uv itself (BACKLOG 50); None when written, else
    why it could not be."""
    import shutil

    uv = shutil.which("uv")
    if uv is None:
        return "uv is not on PATH to export uv.lock"
    if not (REPO_ROOT / "uv.lock").is_file():
        return f"there is no uv.lock in {REPO_ROOT}"
    try:
        run([uv, "export", "--project", str(REPO_ROOT), "--frozen", "--no-hashes",
             "--no-emit-project", "--no-header", "--quiet", "--format", "requirements-txt",
             "--output-file", str(destination)], check=True)
    except (OSError, subprocess.CalledProcessError):
        return "uv could not export uv.lock"
    return None


def pip_install(spec: str, name: str, reinstall: bool = False, no_deps: bool = False) -> bool:
    """Install an optional accelerator with pip, **never replacing a locked
    package** (BACKLOG 50). pip is not in a deploy-created .venv, but the ARM
    source build puts it there, and an accelerator installed with its
    dependencies can ask for another torch. The locked set goes in as
    constraints, so pip refuses such an install instead of making it; when the
    constraints cannot be had, nothing is installed.

    ``spec`` is the table's install spec (`choose_spec`): a requirement and
    pip's options for it, such as "flash-attn --no-build-isolation". It is
    split here, the one place a spec is parsed, so each part is an argument of
    its own: given whole, pip read the string as one invalid requirement
    (the recheck of d4209c4, OPS-RECHECK-I1)."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="shepherd-lock-") as scratch:
        constraints = Path(scratch) / "constraints.txt"
        why = export_lock_constraints(constraints)
        if why is not None:
            log(f"Not installing {name}: {why}, so the install could replace a locked "
                "package such as torch")
            return False
        args = [sys.executable, "-m", "pip", "install", "-U", "-c", str(constraints)]
        if reinstall:
            args.extend(["--force-reinstall", "-I"])
        if no_deps:
            args.append("--no-deps")
        args.extend(shlex.split(spec))
        log(f"Installing {name}: {spec} (constrained to uv.lock)")
        try:
            run(args, check=True)
            return True
        except subprocess.CalledProcessError:
            log(f"WARNING: pip install failed for {name}; if it needs a version uv.lock "
                "does not hold, the locked set was kept and it is not installed")
            return False

class Accelerator(NamedTuple):
    """An optional accelerator the launcher can install: its key in an
    attention order (and its flag's ``args`` name), the module it imports
    as, its name in the log, the flag that asks a start to install it,
    whether it runs on ARM, and whether pip installs it without its
    dependencies."""
    key: str
    module: str
    display: str
    flag: str
    on_arm: bool = True
    no_deps: bool = False


#: In the order a start considers them; an accelerator is installed only when
#: its flag asks for it (`main`).
ACCELERATORS = (
    Accelerator("flash_attn", "flash_attn", "FlashAttention", "--flash-attn", on_arm=False,
                no_deps=True),
    Accelerator("xformers", "xformers", "xFormers", "--xformers"),
    Accelerator("sage_attn", "sage_attention", "SageAttention", "--sage-attn"),
)


def have_module(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False

def choose_spec(table: Dict[str, Any], key: str, *, torch_v: Optional[str], cuda_v: Optional[str], py_v: str,
                os_name: str, arch: str) -> Optional[str]:
    entry = table.get(key) or {}
    try:
        os_map = entry.get(os_name, {})
        arch_map = os_map.get(arch, {})
        py_map = arch_map.get(py_v, {})
        if torch_v and cuda_v and f"{torch_v}+cu{cuda_v.replace('.', '')}" in py_map:
            return py_map[f"{torch_v}+cu{cuda_v.replace('.', '')}"]
        if "*" in py_map:
            return py_map["*"]
        return entry.get("default")
    except Exception:
        return entry.get("default")

# ---------------------------------------------------------------------------
# The local start (UI-06): a start without a terminal, its log and error page
# ---------------------------------------------------------------------------
def has_terminal() -> bool:
    """Whether any of this process's standard streams is a terminal. A start
    from the desktop entry has none (its output would be lost), a start from
    a terminal has one even when its output is piped."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            if stream is not None and stream.isatty():
                return True
        except (AttributeError, ValueError, OSError):
            continue
    return False


def output_is_kept() -> bool:
    """Whether this process's output already goes somewhere a person chose:
    a regular file or a pipe (``nohup ... > my.log``, a script or a test that
    reads it). A desktop start's output goes to /dev/null or to the session's
    journal socket, where nobody reads it; that, or a closed descriptor, is
    not kept."""
    import stat

    try:
        mode = os.fstat(1).st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) or stat.S_ISFIFO(mode)


def send_output_to_log(path: Path) -> None:
    """Append this process's output, and the server's it execs into, to the
    launch log at `path` (0600; moved aside once it is large,
    `launch_instance.rotate_log`)."""
    ensure_directory(path.parent)
    instances.rotate_log(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start of SHEPHERD-Advanced "
          f"(pid {os.getpid()}, {REPO_ROOT}) =====", flush=True)


#: The error page's title and heading, which say what happened: a start that
#: failed (the default), or a second start that opened nothing because the
#: service of its configuration runs without what it sets (`open_running`).
DID_NOT_START = "SHEPHERD-Advanced did not start"
RUNS_WITHOUT_SETTINGS = "SHEPHERD-Advanced already runs without the settings this start sets"

ERROR_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{heading}</title>
<style>
body {{ font: 15px/1.5 system-ui, sans-serif; color: #1f2430; background: #fff;
       max-width: 720px; margin: 40px auto; padding: 0 16px; }}
h1 {{ font-size: 20px; font-weight: 600; }}
dt {{ font-size: 14px; font-weight: 600; margin-top: 16px; }}
dd {{ margin: 4px 0 0; }}
code {{ font-size: 14px; word-break: break-all; }}
.meta {{ font-size: 14px; color: #434a58; margin-top: 24px; }}
</style></head>
<body><h1>{heading}</h1>
<dl><dt>What failed</dt><dd>{what}</dd>
<dt>What to do</dt><dd>{remedy}</dd>
<dt>The full log</dt><dd><code>{log}</code></dd></dl>
<p class="meta">Written {when} by the launcher of {repo}.</p>
</body></html>
"""


def write_error_page(page: Path, what: str, remedy: str, log_path: Optional[Path],
                     heading: str = DID_NOT_START) -> Path:
    """The local error page of a start without a terminal: under `heading`
    (its title too), what failed, what to do and where the log is, every
    value escaped. Written whole (0600), by the settings files' one atomic
    writer (`write_text_atomic`)."""
    text = ERROR_PAGE.format(heading=html.escape(heading),
                             what=html.escape(what), remedy=html.escape(remedy),
                             log=html.escape(str(log_path) if log_path else "none was written"),
                             when=html.escape(time.strftime("%Y-%m-%d %H:%M:%S")),
                             repo=html.escape(str(REPO_ROOT)))
    write_text_atomic(page, text, file_mode=0o600)
    return page


def show_error_page(page: Path, what: str, remedy: str, log_path: Optional[Path],
                    heading: str = DID_NOT_START) -> None:
    """Write the error page and open it in the browser; a failure to do
    either is logged, never raised (the start has failed already)."""
    try:
        written = write_error_page(page, what, remedy, log_path, heading)
        log(f"The error page is {written}")
        open_in_browser(written.as_uri())
    except Exception as exc:  # noqa: BLE001 - the start has failed already
        log(f"WARNING: the error page could not be shown ({exc})")


#: What to do when the server ended before it answered.
SERVER_GONE = "The service stopped before it was ready to show the dashboard."
SERVER_GONE_REMEDY = ("Read the end of the log for the reason, correct it, then start "
                      "SHEPHERD-Advanced again.")


def server_gone_page(page: str, log_path: str) -> Callable[[], None]:
    """The browser opener's `on_gone` for a start without a terminal: the
    error page saying the server ended before it answered."""
    return lambda: show_error_page(Path(page), SERVER_GONE, SERVER_GONE_REMEDY,
                                   Path(log_path) if log_path else None)


class Failure:
    """How this start reports a failure before the server is up: a log line
    always, and the error page too when it has no terminal (`page` set),
    under the heading that says what happened (`DID_NOT_START` unless the
    failure names another)."""

    def __init__(self, page: Optional[Path] = None, log_path: Optional[Path] = None) -> None:
        self.page = page
        self.log_path = log_path

    def __call__(self, what: str, remedy: str, status: int,
                 heading: str = DID_NOT_START) -> int:
        log(f"ERROR: {what}")
        if self.page is not None:
            show_error_page(self.page, what, remedy, self.log_path, heading)
        return status


# ---------------------------------------------------------------------------
# The port (plan §4.5)
# ---------------------------------------------------------------------------
#: `port_problem`'s answer for a port another socket holds.
IN_USE = "in use by another program"


def port_problem(host: str, port: int) -> Optional[str]:
    """None when uvicorn could bind `host`:`port` now, else why not
    (`IN_USE`, or the system's reason). Bound as uvicorn binds it: the
    address family of the host, and on POSIX ``SO_REUSEADDR``, so a port
    left in TIME_WAIT is free; on Windows, where that option would let a
    second socket take a port in use, without it."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
    except OSError as exc:
        return exc.strerror or str(exc)
    with sock:
        if os.name == "posix":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host.strip("[]"), port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE or getattr(exc, "winerror", None) == 10048:
                return IN_USE
            return exc.strerror or str(exc)
    return None


class PortChoice(NamedTuple):
    """`choose_port`'s answer: the port to serve on (None when the
    arguments' port is left to uvicorn, or nothing is started), a note to
    log, and the error that starts nothing."""

    port: Optional[int]
    note: Optional[str] = None
    error: Optional[str] = None


def choose_port(host: str, requested: Optional[str],
                problem: Optional[Callable[[str, int], Optional[str]]] = None) -> PortChoice:
    """The port this start serves on (plan §4.5). A port passed with
    ``--port`` is that port or nothing: in use, the start is refused with the
    reason. Without one, `DEFAULT_PORT`, else the first free one of
    `FALLBACK_PORTS`, said in the note. A ``--port`` that is not a number, or
    0 (any free port, which no URL can name in advance), is left to uvicorn.
    `problem` is `port_problem` unless given."""
    problem = problem or port_problem
    last = FALLBACK_PORTS[-1]
    if requested is not None:
        try:
            port = int(requested)
        except ValueError:
            return PortChoice(None)
        if port <= 0:
            return PortChoice(None)
        why = problem(host, port)
        if why is None:
            return PortChoice(port)
        if why == IN_USE:
            return PortChoice(None, error=(
                f"Port {port}, passed with --port, is in use by another program, so nothing "
                f"was started. Pass a free port, or leave --port out and the launcher takes "
                f"a free one from {DEFAULT_PORT} to {last}."))
        return PortChoice(None, error=f"Port {port} on {host} cannot be used ({why}), so "
                                      "nothing was started.")
    for port in (DEFAULT_PORT, *FALLBACK_PORTS):
        why = problem(host, port)
        if why is None:
            if port == DEFAULT_PORT:
                return PortChoice(port)
            taken = (f"Port {DEFAULT_PORT} is" if port == DEFAULT_PORT + 1
                     else f"Ports {DEFAULT_PORT} to {port - 1} are")
            return PortChoice(port, note=f"{taken} in use by other programs, so the service "
                                         f"uses port {port}.")
        if why != IN_USE:
            return PortChoice(None, error=f"Port {port} on {host} cannot be used ({why}), so "
                                          "nothing was started.")
    return PortChoice(None, error=(
        f"Ports {DEFAULT_PORT} to {last} are all in use by other programs, so nothing was "
        "started. Close one of them, or pass a free port: launch_shepherd.sh -- --port N"))


# ---------------------------------------------------------------------------
# One instance per configuration (plan §4.5; src/utils/launch_instance.py)
# ---------------------------------------------------------------------------
def instance_answers(info: Mapping[str, Any], timeout: float = 3.0) -> bool:
    """Whether the instance an info file describes answers ``/health`` with
    the file's own id; a 200 alone never counts."""
    url = info.get("url")
    if not isinstance(url, str) or not url.startswith("http://"):
        return False
    try:
        return _answers_as(urlopen(f"{url}/health", timeout=timeout), info.get("instance_id"))
    except (URLError, OSError, ValueError):
        return False


class Claim(NamedTuple):
    """What `claim_instance` found: this start's lock (`fd`), or the info of
    the instance of this configuration that runs and answers (`running`)."""

    fd: Optional[int]
    running: Optional[Dict[str, Any]] = None


def claim_instance(files: "instances.Instance", progress_every: float = 30.0) -> Claim:
    """Take this configuration's instance lock, or find the instance that
    holds it. While another start holds the lock and does not answer yet (it
    is still starting), wait: it answers, and this start opens it, or it ends
    and releases the lock, and this start takes it. No deadline, as the
    browser opener has none: a start that loads a large table takes minutes.

    Raises:
        OSError: the lock file cannot be created or opened."""
    started = time.monotonic()
    next_progress = 0.0
    while True:
        fd = instances.claim(files.lock)
        if fd is not None:
            return Claim(fd)
        info = instances.read_info(files.info)
        if info is not None and instance_answers(info):
            return Claim(None, info)
        elapsed = time.monotonic() - started
        if elapsed >= next_progress:
            log(f"Another start of this configuration is under way ({elapsed:.0f} s; its lock "
                f"is {files.lock}); waiting for it to answer, or to end")
            next_progress = elapsed + progress_every
        time.sleep(1.0)


def operator_controls_words(environ: Mapping[str, str]) -> str:
    """The "Operator controls" line's words for a service started with
    `environ`: on, and why, or the service's own reason they are off, in its
    parts (the state, then why, in whole sentences; T08 flow: a cut sentence
    went on in lower case)."""
    mode = deployment_mode(environ)
    if mode.available:
        return ("on for requests made on this computer "
                + ("(the local start)" if environ.get(ENV_LOCAL_ENTRY) == LOCAL_ENTRY
                   and mode.source != "environment" else "(the deployment mode set at start)"))
    return f"{MODE_OFF_STATE}. {mode_off_why(mode)}"


#: Why a second start that asks for other settings opens nothing, and what to do.
OTHER_SETTINGS_REMEDY = (
    "To start with what this start sets, stop the running service first (Settings › Stop "
    "service in it, Ctrl-C in its terminal, or end pid {pid}), then start again. To open the "
    "running service as it is, start without those settings.")


def open_running(running: Mapping[str, Any], args: argparse.Namespace, fail: Failure) -> int:
    """A second start of a configuration that runs (plan §4.5): open it, or,
    when it does not have what this start asks for, open nothing and say what
    runs (OPS-C-R1, `launch_instance.asks_not_met`), under the error page's
    `RUNS_WITHOUT_SETTINGS` heading, never "did not start". Its "Operator
    controls" line is the running service's, never this start's own."""
    ui = f"{running['url']}/ui"
    pid = running.get("pid")
    unmet = instances.asks_not_met(running, os.environ)
    environ = instances.running_environment(running, os.environ)
    if unmet:
        # What runs: its operator controls always, and each location asked
        # for that it does not have; then what this start sets.
        if environ is None:
            runs = f"its settings {instances.NOT_RECORDED}"
        else:
            runs = ", ".join(
                ["operator controls " + ("on" if deployment_mode(environ).available else "off")]
                + [f"{ask.name} {ask.running}" for ask in unmet
                   if ask.variable != ENV_DEPLOYMENT_MODE])
        asked = ", ".join(f"{ask.name} {ask.asked}" for ask in unmet)
        refs = ", ".join(ask.variable for ask in unmet)
        remedy = OTHER_SETTINGS_REMEDY.format(pid=pid)
        status = fail(f"SHEPHERD-Advanced already runs for this configuration at {ui} (pid {pid}) "
                      "without the settings this start sets, so it was not opened and no second "
                      f"backend is started on the same data. Running: {runs}. This start sets: "
                      f"{asked} (Ref: {refs}).", remedy, 1, heading=RUNS_WITHOUT_SETTINGS)
        log(remedy)
        return status
    log(f"SHEPHERD-Advanced already runs for this configuration at {ui} (pid {pid}); no second "
        "backend is started on the same data.")
    if args.local_entry:
        log("Operator controls     : " + (operator_controls_words(environ) if environ is not None
                                          else instances.NOT_RECORDED + "; the dashboard's "
                                          "header says them"))
    if not args.no_browser:
        open_in_browser(ui)
    return 0


# ---------------------------------------------------------------------------
# The saved settings file this start uses (--settings-file)
# ---------------------------------------------------------------------------
def apply_settings_file(argument: Optional[str], cwd: Path, environ) -> None:
    """``--settings-file PATH``: the saved settings file the service reads
    (`SHEPHERD_SETTINGS_FILE`; an A/B slot), over one the environment names.
    A relative path is read against the directory the launcher was started
    in, before it changes to the repository."""
    if argument is None:
        return
    path = Path(argument).expanduser()
    if not path.is_absolute():
        path = cwd / path
    environ[ENV_SETTINGS_FILE] = os.path.abspath(path)


def settings_file_words(environ, named_here: bool) -> str:
    """Which saved settings file the service reads, and why, for the log."""
    location = settings_location(environ)
    if location.problem is not None:
        return f"WARNING: {location.problem}; the data locations come only from the environment"
    if location.path is None:
        return ("none (HOME is not set): the data locations come only from the environment")
    why = ("--settings-file" if named_here
           else ENV_SETTINGS_FILE if location.explicit else "this account's")
    return f"{location.path} ({why})"


# ---------------------------------------------------------------------------
# The desktop entry (UI-06e)
# ---------------------------------------------------------------------------
#: Characters the desktop entry specification reserves in an Exec argument.
_EXEC_RESERVED = set(' \t\n"\'\\><~|&;$*?#()`')


def desktop_exec_argument(argument: str) -> str:
    """`argument` as one Exec argument of a desktop entry: quoted when it holds
    a reserved character (``"``, `` ` ``, ``$`` and ``\\`` escaped inside the
    quotes), then every backslash doubled (the value is also a string whose
    ``\\`` is an escape) and ``%`` written ``%%`` (a field code otherwise)."""
    if "\n" in argument or "\r" in argument:
        raise ValueError(f"a path with a line break cannot be started from a desktop entry: "
                         f"{argument!r}")
    if any(ch in _EXEC_RESERVED for ch in argument):
        inner = "".join("\\" + ch if ch in '"`$\\' else ch for ch in argument)
        argument = f'"{inner}"'
    return argument.replace("\\", "\\\\").replace("%", "%%")


def _desktop_string(value: str) -> str:
    """A desktop entry string value: its backslashes escaped; a line break
    refused (`desktop_exec_argument` refuses it in a path first)."""
    if "\n" in value or "\r" in value:
        raise ValueError(f"a desktop entry value cannot hold a line break: {value!r}")
    return value.replace("\\", "\\\\")


def desktop_entry_text(settings_file: Optional[str]) -> str:
    """The desktop entry of this checkout: `LAUNCH_SCRIPT` by bash (so it
    starts even where an unpacked archive dropped the script's exec bit), with
    ``--settings-file`` for an A/B slot, no terminal, this repository's icon."""
    argv = ["bash", str(LAUNCH_SCRIPT)]
    name = "SHEPHERD-Advanced"
    comment = f"Open the SHEPHERD-Advanced dashboard of {REPO_ROOT} on this computer"
    if settings_file is not None:
        argv += ["--settings-file", settings_file]
        name = f"SHEPHERD-Advanced ({Path(settings_file).parent.name or settings_file})"
        comment += f", with the saved settings {settings_file}"
    exec_line = " ".join(desktop_exec_argument(arg) for arg in argv)
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        "Version=1.5",
        f"Name={_desktop_string(name)}",
        f"Comment={_desktop_string(comment)}",
        f"Exec={exec_line}",
        f"Path={_desktop_string(str(REPO_ROOT))}",
        f"Icon={_desktop_string(str(DESKTOP_ICON))}",
        "Terminal=false",
        "StartupNotify=false",
        "Categories=Science;MedicalSoftware;",
    ]
    return "\n".join(lines) + "\n"


def desktop_entry_path(environ, settings_file: Optional[str]) -> Optional[Path]:
    """Where the entry goes: ``${XDG_DATA_HOME:-~/.local/share}/applications``,
    ``shepherd-advanced.desktop``, or for a settings file of its own one named
    after it, so an A slot's and a B slot's entries are both kept. None when
    HOME is not set."""
    data = environ.get("XDG_DATA_HOME") or ""
    base = Path(data) if data and Path(data).is_absolute() else None
    if base is None:
        home = environ.get("HOME") or ""
        if not home or not Path(home).is_absolute():
            return None
        base = Path(home) / ".local" / "share"
    name = "shepherd-advanced.desktop"
    if settings_file is not None:
        name = f"shepherd-advanced-{instances.instance_key(REPO_ROOT, settings_file)[:8]}.desktop"
    return base / "applications" / name


def install_desktop_entry(environ, settings_file: Optional[str]) -> int:
    """``--install-desktop-entry``: write this checkout's desktop entry
    (`desktop_entry_text`) and say where, with the paths it names."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        log("ERROR: a desktop entry is a Linux desktop's; on Windows start launch_shepherd.cmd "
            "by double click, or make a shortcut to it")
        return 2
    target = desktop_entry_path(environ, settings_file)
    if target is None:
        log("ERROR: HOME is not set, so where the desktop entry goes cannot be told")
        return 2
    try:
        text = desktop_entry_text(settings_file)
    except ValueError as exc:
        log(f"ERROR: {exc}")
        return 2
    # Whole, by the settings files' one atomic writer (`write_text_atomic`).
    write_text_atomic(target, text, file_mode=0o644, directory_mode=0o755)
    log(f"Desktop entry written: {target}")
    log(f"  It starts      : {LAUNCH_SCRIPT}")
    log(f"  Settings file  : {settings_file or 'this account’s (the default)'}")
    log(f"  Icon           : {DESKTOP_ICON}")
    if not (REPO_ROOT / ".venv").is_dir():
        log(f"WARNING: {REPO_ROOT / '.venv'} is not there yet: run deploy.sh first, or the "
            "entry opens a page saying so")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shep_launch", description="SHEPHERD launcher for optional accelerators", formatter_class=argparse.RawTextHelpFormatter,
                                epilog="Arguments after -- go to the main app unchanged, e.g.  shep_launch.py --no-browser -- --port 8264")
    g = p.add_argument_group("Accelerator flags (opt-in like SD WebUI)",
                             "A start installs an accelerator only when its own flag asks for it;\n"
                             "an attention order uses the installed ones and installs none.")
    g.add_argument("--flash-attn", action="store_true", help="Use FlashAttention, installing it when missing (Windows/x86; skipped on ARM)")
    g.add_argument("--xformers", action="store_true", help="Use xFormers memory-efficient attention, installing it when missing")
    g.add_argument("--sage-attn", action="store_true", help="Use SageAttention (community plugin), installing it when missing")
    g.add_argument("--cudnn-sdpa", action="store_true", help="Prefer cuDNN SDPA (default if nothing else selected)")
    g.add_argument("--torch-sdpa", action="store_true", help="Prefer vanilla Torch SDPA")
    g.add_argument("--naive-attn", action="store_true", help="Force naive/matmul attention as last resort")
    r = p.add_argument_group("Reinstall/behavior controls")
    r.add_argument("--reinstall-flash-attn", action="store_true", help="With --flash-attn: its install, when missing, uses pip --force-reinstall")
    r.add_argument("--reinstall-xformers", action="store_true", help="With --xformers: its install, when missing, uses pip --force-reinstall")
    r.add_argument("--reinstall-sage-attn", action="store_true", help="With --sage-attn: its install, when missing, uses pip --force-reinstall")
    r.add_argument("--no-auto-install", action="store_true",
                   help="Install nothing: --flash-attn, --xformers and --sage-attn use their\n"
                        "accelerator only when it is installed")
    o = p.add_argument_group("Ordering and plugins")
    o.add_argument("--attention-order", type=str, default="", help="Comma list, e.g. flash_attn,xformers,cudnn_sdpa,torch_sdpa,naive\n(uses what is installed; installs nothing)")
    o.add_argument("--plugin", type=str, default="", help="Custom plugin path 'module.sub:Class' (will be checked)")
    d = p.add_argument_group("Diagnostics")
    d.add_argument("--print-plan", action="store_true", help="Print chosen plan and exit code")
    d.add_argument("--dry-run", action="store_true", help="Simulate actions without installing or launching")
    d.add_argument("--skip-launch", action="store_true", help="Do not start main app after setup")
    m = p.add_argument_group("Main app")
    m.add_argument("--entry", type=str, default=DEFAULT_ENTRY, help="Python module to run with -m (default: uvicorn)")
    m.add_argument("--no-browser", action="store_true", help="Do not open a browser once the server is ready.")
    s = p.add_argument_group("Local start")
    s.add_argument("--local-entry", action="store_true",
                   help="Started from the desktop entry or launch_shepherd.sh/.cmd. On a loopback\n"
                        "bind the operator controls are on for this computer unless\n"
                        "SHEPHERD_DEPLOYMENT_MODE says otherwise; without a terminal the output\n"
                        "goes to a log file and a failed start opens a page saying why.")
    s.add_argument("--settings-file", metavar="PATH",
                   help="The saved settings file the service reads (an A/B slot); default:\n"
                        "SHEPHERD_SETTINGS_FILE, else this account's.")
    s.add_argument("--install-desktop-entry", action="store_true",
                   help="Write a desktop entry that starts this checkout (with --settings-file,\n"
                        "with that file) and exit.")
    return p

def split_passthrough(argv: List[str]) -> Tuple[List[str], List[str]]:
    """Split at the first ``--``: the launcher's arguments, then the app's.

    argparse cannot do this itself. It reads ``--`` as the end of options, so
    an option *named* ``--`` is never matched, and with no positional to take
    the remainder it refused ``-- --port 8264`` as unrecognized arguments.
    """
    if "--" in argv:
        split = argv.index("--")
        return argv[:split], argv[split + 1:]
    return argv, []
    
def last_option(uvicorn_args: List[str], name: str) -> Optional[str]:
    """The value of the last ``name`` option in these arguments, in either
    spelling -- ``--port 8264`` or ``--port=8264`` -- as uvicorn takes a
    repeated option; `None` without one."""
    value = None
    for i, arg in enumerate(uvicorn_args):
        option, sep, inline = arg.partition("=")
        if option != name:
            continue
        if sep:
            value = inline
        elif i + 1 < len(uvicorn_args):
            value = uvicorn_args[i + 1]
    return value


def bind_address(uvicorn_args: List[str]) -> Tuple[str, str]:
    """The host and port uvicorn binds with these arguments (`last_option`).
    Without either, uvicorn's own defaults."""
    host = last_option(uvicorn_args, "--host")
    port = last_option(uvicorn_args, "--port")
    return ("127.0.0.1" if host is None else host), ("8000" if port is None else port)


def socket_option(uvicorn_args: List[str], environ: Mapping[str, str], name: str) -> Optional[str]:
    """uvicorn's ``--fd`` or ``--uds`` as it reads one: from its command line,
    else from ``UVICORN_FD`` / ``UVICORN_UDS`` (its options' environment
    variables, click's ``auto_envvar_prefix``; an empty value counts as unset).
    `None` when neither gives it."""
    value = last_option(uvicorn_args, f"--{name}")
    if value is None:
        value = environ.get(f"UVICORN_{name.upper()}") or None
    return value


class BindPlan(NamedTuple):
    """What uvicorn will listen on, and SPEC_4 §3.1's decision for it."""

    #: The bind guard's decision (`allowed`, `message`); `None` for a unix
    #: socket, which is not a network bind.
    decision: Any
    #: What the "Listening on" line says.
    listening: str
    #: The host passed to application startup as ``SHEPHERD_BIND_HOST`` and
    #: shown as URLs; `None` when uvicorn binds no host.
    host: Optional[str]


def bind_plan(uvicorn_args: List[str], environ: Mapping[str, str]) -> BindPlan:
    """What uvicorn listens on with these arguments and this environment:
    a socket it is handed (``--fd``), else a unix socket (``--uds``), else
    ``--host`` and ``--port``. (One server process takes ``--fd`` first; with
    ``--reload`` or ``--workers`` uvicorn takes ``--uds`` first, and the
    stricter ``--fd`` rule is applied whenever one is given.)

    - ``--fd``: an inherited socket, whose address this launcher does not
      read: a network bind (`inherited_socket_bind`), refused without
      ``SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1``;
    - ``--uds``: a unix socket, reachable on this machine only, and every
      operator control is refused through it (D8 sees no loopback address);
    - otherwise the host (`network_bind`).
    """
    fd = socket_option(uvicorn_args, environ, "fd")
    if fd is not None:
        decision = inherited_socket_bind(fd, environ)
        return BindPlan(decision, f"{decision.host}, whose address is not read here", None)
    uds = socket_option(uvicorn_args, environ, "uds")
    if uds is not None:
        return BindPlan(None, (f"unix socket {uds} (this machine only; the operator controls "
                               "are refused through it, D8)"), None)
    host, port = bind_address(uvicorn_args)
    reach = ("loopback: this machine only" if bind_is_loopback(host)
             else "all interfaces" if host.strip() in ("", "0.0.0.0", "::")
             else "this address")
    shown = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return BindPlan(network_bind(host, environ), f"{shown}:{port} ({reach})", host)


def display_base_url(uvicorn_args: List[str]) -> str:
    """The URL to show for a server started with these uvicorn arguments.

    The address is `bind_address`'s. A wildcard host is shown as the matching
    loopback address, which a browser on this machine can open -- an empty
    ``--host=`` binds all interfaces too, so it counts as one. An IPv6 literal
    is bracketed, as a URL requires, unless it already is.
    """
    host, port = bind_address(uvicorn_args)
    host = {"": "127.0.0.1", "0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def can_hand_over() -> bool:
    """Whether this process may become the server by ``exec``, as it then does
    on every start, a browser to open or not.

    POSIX only. There ``execve`` replaces the process image and keeps the PID,
    so whoever started the launcher -- a terminal, launch_shepherd.sh, a
    service manager -- holds the server's PID and its signals reach the
    server. On Windows ``os.exec*`` starts a new process and ends this one:
    the PID changes and a supervisor loses track, so the launcher waits
    instead (`serve`).
    """
    return os.name == "posix"

def exec_server(cmd: List[str]) -> int:
    """Replace this process with ``cmd``; return a non-zero status only on failure.

    Output is flushed first, because ``exec`` discards whatever is still buffered.
    """
    log("$ " + " ".join(cmd))
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.execve(cmd[0], cmd, os.environ)
    except OSError as exc:
        log(f"ERROR: could not start the server ({cmd[0]}): {exc}")
    return 1

def collect_args_from_env_and_cli() -> Tuple[List[str], List[str]]:
    """The launcher's arguments and the app's, from the environment and the CLI.

    **Each source is split at its own ``--`` before the two are joined.**
    Joining first let a ``--`` in the environment turn every launcher flag on
    the command line -- ``--no-auto-install``, ``--no-browser`` -- into app
    arguments: the launcher then installed and opened a browser it had been told
    not to, and uvicorn refused flags it does not know. Within each half the
    environment comes first, so for the app, whose repeated options take the
    last value, the command line wins. ``SHEP_COMMANDLINE_ARGS`` still takes
    precedence over ``COMMANDLINE_ARGS``; only one of them is read.
    """
    env = os.getenv("SHEP_COMMANDLINE_ARGS") or os.getenv("COMMANDLINE_ARGS") or ""
    env_own, env_app = split_passthrough(shlex.split(env))
    cli_own, cli_app = split_passthrough(sys.argv[1:])
    return env_own + cli_own, env_app + cli_app

def enter_repository() -> None:
    """Change to the repository root, whatever directory the launcher was
    started in (UI-06a): uvicorn imports ``src`` from its working directory,
    and Restart Backend re-execs the server in it."""
    os.chdir(REPO_ROOT)


def _drop_instance(info_path: Optional[Path], instance_id: Optional[str],
                   fd: Optional[int]) -> None:
    """A start that ends before its server is up: its info file removed (when
    it still names this start) and its instance lock released."""
    if info_path is not None and instance_id is not None:
        info = instances.read_info(info_path)
        if info is not None and info.get("instance_id") == instance_id:
            try:
                info_path.unlink()
            except OSError:
                pass
    if fd is not None:
        os.close(fd)


#: What to do after a refused bind, said on the error page.
BIND_REMEDY = ("Start it on this computer's own address (leave --host out), or accept the risk "
               "as the deployment guide's section 4.13 says.")
#: What to do when the port cannot be had.
PORT_REMEDY = ("Close the program that uses the port, or start SHEPHERD-Advanced on another "
               "port; leave --port out and the launcher picks a free one.")
#: What to do when the server program cannot be started at all.
EXEC_REMEDY = "Run deploy.sh again in the repository, then start SHEPHERD-Advanced again."


def main() -> int:
    parser = build_parser()
    own_args, passthrough_args = collect_args_from_env_and_cli()
    args = parser.parse_args(own_args)
    args.passthrough = passthrough_args

    # Collect passthrough args
    passthrough: List[str] = []
    if args.passthrough:
        passthrough = list(args.passthrough)
        if passthrough and passthrough[0] == "--":
            passthrough = passthrough[1:]

    # **The launcher's own variables are this start's** (`LAUNCHER_VARIABLES`):
    # an inherited bind host, local start mark or instance never reaches the
    # server; each is set again below only when this start decides it.
    for name in LAUNCHER_VARIABLES:
        os.environ.pop(name, None)
    try:
        started_in = Path.cwd()
    except OSError:  # the directory it was started in is gone
        started_in = REPO_ROOT
    apply_settings_file(args.settings_file, started_in, os.environ)
    if args.install_desktop_entry:
        if args.settings_file is None and os.environ.get(ENV_SETTINGS_FILE):
            log(f"WARNING: {ENV_SETTINGS_FILE} is set in this shell, but a desktop entry starts "
                "without this shell's environment; pass --settings-file to give the entry "
                "that file")
        return install_desktop_entry(
            os.environ, os.environ[ENV_SETTINGS_FILE] if args.settings_file else None)
    # **Any working directory** (UI-06a), before anything is read or started.
    enter_repository()

    launching = not (args.print_plan or args.dry_run or args.skip_launch)
    state_dir = instances.state_directory(os.environ)
    # **No terminal** (the desktop entry; UI-06e): a failure before the server
    # is up opens a local page saying what failed and where the log is, and
    # the output, the server's included, goes to the launch log unless it
    # already goes to a file or a pipe someone chose (`output_is_kept`: a
    # redirect of launch_shepherd.sh keeps working as before).
    log_path: Optional[Path] = None
    page: Optional[Path] = None
    if args.local_entry and launching and not has_terminal() and state_dir is not None:
        page = instances.error_page(state_dir)
        if not output_is_kept():
            try:
                send_output_to_log(instances.log_file(state_dir))
                log_path = instances.log_file(state_dir)
            except OSError as exc:
                log(f"WARNING: the launch log cannot be written ({exc}); the output stays "
                    "where it was")
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
    fail = Failure(page, log_path)
    log(f"Saved settings file   : {settings_file_words(os.environ, args.settings_file is not None)}")

    # **The bind guard** (SPEC_4 §3.1), before anything is installed or
    # started: what uvicorn will listen on (`bind_plan`) other than loopback is
    # refused unless SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1 accepts the
    # risk (no control: D8 still refuses operator controls off loopback). Only
    # for the uvicorn entry, whose options this launcher can read. The host is
    # passed to application startup, which checks it again; no host, none.
    bind: Optional[BindPlan] = None
    if args.entry == "uvicorn":
        bind = bind_plan(UVICORN_DEFAULT_ARGS + passthrough, os.environ)
        if bind.decision is not None and not bind.decision.allowed:
            return fail(bind.decision.message, BIND_REMEDY, 2)
        if bind.host is not None:
            os.environ[ENV_BIND_HOST] = bind.host

    # **The local start's mode** (D06): ``--local-entry`` on a loopback bind
    # marks the start, and an unset deployment mode is then M1 for this
    # computer (`server_settings.deployment_mode`); an explicit one decides,
    # and D8 still checks every request. Its "Operator controls" line is said
    # once this start knows which service the user gets: this one, or the one
    # that already runs (`open_running`).
    if args.local_entry:
        if bind is not None and bind.host is not None and bind_is_loopback(bind.host):
            os.environ[ENV_LOCAL_ENTRY] = LOCAL_ENTRY

    # **One instance per configuration, and the port** (plan §4.5), before
    # anything is installed: a start of a configuration that already runs
    # opens that one and ends, unless it sets a deployment mode or a data
    # location that one does not have (`open_running`: then it opens nothing
    # and exits non-zero), and a port in use is said before the long part of
    # a start. Only for a server on a host and port (a socket has no URL to
    # tell an instance by).
    uvicorn_defaults = list(UVICORN_DEFAULT_ARGS)
    instance_fd: Optional[int] = None
    info_path: Optional[Path] = None
    instance_id: Optional[str] = None
    settings_file = settings_location(os.environ).path
    if launching and bind is not None and bind.host is not None:
        if state_dir is None:
            log("Single instance       : not checked (HOME is not set, so there is no state "
                "directory)")
        else:
            files = instances.instance_files(state_dir,
                                             instances.instance_key(REPO_ROOT, settings_file))
            try:
                found: Optional[Claim] = claim_instance(files)
            except OSError as exc:
                log(f"WARNING: the single-instance check is skipped: {files.lock} cannot be "
                    f"used ({exc})")
                found = None
            if found is not None and found.running is not None:
                return open_running(found.running, args, fail)
            if found is not None:
                instance_fd, info_path = found.fd, files.info
                # The lock was free: an info file still there is what a
                # crash left (FL-SVC), said once before it is replaced.
                left = instances.read_info(files.info)
                if left is not None:
                    log(f"WARNING: {instances.ended_without_stop_words(left)}")
        port_requested = last_option(passthrough, "--port")
        choice = choose_port(bind.host, port_requested)
        if choice.error is not None:
            _drop_instance(None, None, instance_fd)
            return fail(choice.error, PORT_REMEDY, 1)
        if choice.note is not None:
            log(f"WARNING: {choice.note}")
        if choice.port is not None and port_requested is None:
            uvicorn_defaults[uvicorn_defaults.index("--port") + 1] = str(choice.port)
        if choice.port is None and instance_fd is not None:
            log("Single instance       : not recorded (the port is uvicorn's to choose, so no "
                "URL names this start)")
            _drop_instance(None, None, instance_fd)
            instance_fd = info_path = None
    if args.local_entry:
        log(f"Operator controls     : {operator_controls_words(os.environ)}")
    base_url = display_base_url(uvicorn_defaults + passthrough)
    if instance_fd is not None and info_path is not None:
        instance_id = instances.new_instance_id()
        try:
            instances.write_info(info_path, instance_id=instance_id, pid=os.getpid(),
                                 port=int(bind_address(uvicorn_defaults + passthrough)[1]),
                                 url=base_url, repo_root=REPO_ROOT, settings_file=settings_file,
                                 log=log_path, environ=os.environ)
        except (OSError, ValueError) as exc:
            log(f"WARNING: the single-instance record {info_path} cannot be written ({exc})")
            _drop_instance(None, None, instance_fd)
            instance_fd = info_path = instance_id = None
        else:
            os.environ[instances.ENV_INSTANCE_ID] = instance_id
            os.environ[instances.ENV_INSTANCE_FILE] = str(info_path)

    os_name = "windows" if is_windows() else ("linux" if sys.platform.startswith("linux") else sys.platform)
    arch = platform.machine().lower()
    py_v = pep440_python()
    torch_v, cuda_v = detect_torch()
    log(f"OS={os_name} arch={arch} py={py_v} torch={torch_v} cuda={cuda_v}")

    # Load deployment config for platform-specific defaults
    deploy_cfg = get_deployment_config()
    platform_key = deploy_cfg.get("_platform", get_platform_key())
    log(f"Platform config: {platform_key}")

    # Get platform-preferred attention backends from deployment.yaml
    attn_cfg = deploy_cfg.get("attention_backend", {})
    platform_prefer = attn_cfg.get("prefer", ["torch_sdpa", "naive"])

    table = read_json(ACCEL_TABLE)
    requested: List[str] = []
    if args.flash_attn: requested.append("flash_attn")
    if args.xformers: requested.append("xformers")
    if args.sage_attn: requested.append("sage_attn")
    if args.torch_sdpa: requested.append("torch_sdpa")
    if args.cudnn_sdpa: requested.append("cudnn_sdpa")
    if args.naive_attn: requested.append("naive")

    if args.attention_order:
        # Explicit order from CLI takes highest priority
        order = [s.strip() for s in args.attention_order.split(",") if s.strip()]
    elif requested:
        # User requested specific accelerators via flags
        seen, order = set(), []
        for k in requested + ["torch_sdpa", "naive"]:
            if k not in seen:
                seen.add(k)
                order.append(k)
    else:
        # Use platform-specific preferences from deployment.yaml
        order = list(platform_prefer)
        # Ensure fallbacks are included
        for fallback in ["torch_sdpa", "naive"]:
            if fallback not in order:
                order.append(fallback)
    # **Use, and install only when asked** (the owner's decision of
    # 2026-11-19). The order -- the platform's preference from
    # deployment.yaml, or --attention-order -- chooses among the accelerators
    # that are installed; it is never permission to install one, with CUDA
    # or without. Only an accelerator's own flag (`ACCELERATORS`) asks this
    # start to install it when it is missing, through `pip_install` and its
    # uv.lock constraints; --no-auto-install turns that install off and
    # --reinstall-<name> adds pip's reinstall to it. A start that does not
    # launch (--print-plan, --skip-launch, --dry-run) installs nothing: it
    # says what the flag would install. An accelerator this start does not use
    # is said in one line, with how to install it, and the start goes on with
    # the next one in the order.
    def usable(accelerator: Accelerator) -> bool:
        if is_arm() and not accelerator.on_arm:
            log(f"Skip {accelerator.display} on ARM architecture"); return False
        if have_module(accelerator.module):
            log(f"{accelerator.display} already available"); return True
        if not getattr(args, accelerator.key):
            log(f"{accelerator.display} is not installed, so this start goes on without it (a "
                f"start installs it only when asked: start once with {accelerator.flag})")
            return False
        if args.no_auto_install:
            log(f"{accelerator.display} is not installed, so this start goes on without it "
                f"(--no-auto-install: {accelerator.flag} uses it only when it is installed)")
            return False
        return install(accelerator)

    def install(accelerator: Accelerator) -> bool:
        spec = choose_spec(table, accelerator.key, torch_v=torch_v, cuda_v=cuda_v, py_v=py_v,
                           os_name=os_name, arch=arch)
        if not spec:
            log(f"No install spec found for {accelerator.display}; skipping"); return False
        if not launching:
            only = "dry-run" if args.dry_run else "print-plan" if args.print_plan else "skip-launch"
            log(f"[{only}] Would install {accelerator.display}: {spec}"); return False
        ok = pip_install(spec, accelerator.display,
                         reinstall=getattr(args, f"reinstall_{accelerator.key}"),
                         no_deps=accelerator.no_deps)
        return ok and have_module(accelerator.module)

    for accelerator in ACCELERATORS:
        if accelerator.key in order and not usable(accelerator):
            order = [x for x in order if x != accelerator.key]
            if accelerator.key == "flash_attn":
                os.environ["FLASHATTN_FORCE_DISABLE"] = "1"
    if args.plugin:
        mod, cls = (args.plugin.split(":", 1) + [""])[:2]
        try:
            __import__(mod)
            os.environ["ATTENTION_PLUGIN"] = args.plugin
            log(f"Plugin available: {args.plugin}")
        except Exception:
            log(f"WARNING: plugin not importable: {args.plugin}")
    os.environ["ATTENTION_ORDER"] = ",".join(order)

    # Export retrieval backend preference if available
    retrieval_cfg = deploy_cfg.get("retrieval_backend", {})
    if retrieval_cfg:
        os.environ["SHEPHERD_RETRIEVAL_BACKEND"] = retrieval_cfg.get("default", "auto")

    # Apply persisted Runtime Settings: CUDA memory allocator preset. Read once
    # here so both this server's in-process CUDA and the training subprocesses it
    # spawns inherit the same allocator. An explicit PYTORCH_ALLOC_CONF /
    # PYTORCH_CUDA_ALLOC_CONF in the environment always wins (e.g. A/B tests).
    # The rule lives in src/config/runtime_presets.py and is the one Restart
    # Backend applies; a malformed settings file falls back to {} and an unknown
    # preset to the default.
    for message in apply_allocator(os.environ, load_runtime_settings()):
        log(message)

    plan = textwrap.dedent(f"""
    === PLAN ===
    Platform              : {platform_key}
    Final attention order : {order}
    Retrieval backend     : {retrieval_cfg.get('default', 'auto')}
    Plugin                : {os.environ.get('ATTENTION_PLUGIN','')}
    FLASHATTN_FORCE_DISAB : {os.environ.get('FLASHATTN_FORCE_DISABLE','')}
    Entry module          : {args.entry}
    App                   : {UVICORN_APP if args.entry == 'uvicorn' else 'N/A'}
    Passthrough args      : {passthrough}
    """)
    print(plan)

    # What the guard above decided, said where the access points are.
    if bind is not None:
        if bind.decision is not None and bind.decision.message:
            log(f"WARNING: {bind.decision.message}")
        log(f"Listening on          : {bind.listening}")
    # A socket has no URL this launcher knows: none is shown, and no browser
    # is opened on one.
    has_urls = bind is None or bind.host is not None
    if has_urls:
        log("Access points (bookmark these):")
        log(f"  Swagger UI (API docs) : {base_url}/docs")
        log(f"  Gradio Dashboard      : {base_url}/ui")
    else:
        log("Access points         : no URL is shown: uvicorn listens on a socket, not on a "
            "host and port this launcher knows")
    if not launching:
        return 0

    # Print web interface endpoints
    if has_urls:
        print(textwrap.dedent(f"""\
        ===  Web Interfaces  ===
          Swagger UI (API docs) : {base_url}/docs
          Gradio Dashboard      : {base_url}/ui
        ========================
        """))

    # Build launch command
    if args.entry == "uvicorn":
        cmd = ([sys.executable, "-m", "uvicorn", UVICORN_APP] + uvicorn_defaults
               + ["--no-access-log"] + passthrough + UVICORN_FORCED_ARGS)
    else:
        cmd = [sys.executable, "-m", args.entry] + passthrough
    # Auto-open browser once server is ready (polls /health until it answers)
    opens_browser = not args.no_browser and has_urls
    if not args.no_browser and not has_urls:
        log("No browser is opened: the server has no URL this launcher knows")
    # **Become the server, on every POSIX start** (round 3 of the review
    # fixes, P3-2). Everything above -- allocator, attention, plugin,
    # retrieval -- is already in os.environ, so exec hands it over whole. No
    # launcher is left resident: the PID that was started is the server's, so
    # a SIGTERM sent to it alone (a service manager, `kill`) is the server's
    # orderly stop, which stops its job; a launcher waiting beside the server
    # was ended by it alone, and the server served on with its job. Restart
    # Backend re-execs it in place as it would a bare uvicorn start. The
    # browser, when one is to be opened, is a helper started first. The
    # instance lock goes through the exec: the server holds it until it ends.
    if can_hand_over():
        if opens_browser:
            start_browser_opener(f"{base_url}/ui", f"{base_url}/health", instance_id=instance_id,
                                 failure_page=page, log_path=log_path)
        if instance_fd is not None:
            instances.hand_to_exec(instance_fd)
        status = exec_server(cmd)
        _drop_instance(info_path, instance_id, instance_fd)
        return fail(f"The server program could not be started ({cmd[0]}).", EXEC_REMEDY, status)
    if opens_browser:
        opener = threading.Thread(
            target=_open_browser_when_ready,
            args=(f"{base_url}/ui", f"{base_url}/health"),
            kwargs={"instance_id": instance_id},
            daemon=True,
        )
        opener.start()

    status = serve(cmd)
    _drop_instance(info_path, instance_id, instance_fd)
    if status != 0:
        return fail(f"The service ended with status {status}.", SERVER_GONE_REMEDY, status)
    return status

if __name__ == "__main__":
    sys.exit(main())