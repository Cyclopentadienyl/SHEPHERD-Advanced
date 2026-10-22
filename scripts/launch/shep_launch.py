# scripts/launch/shep_launch.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys, textwrap, threading, time, webbrowser
from pathlib import Path
from typing import Dict, Any, List, Mapping, NamedTuple, Optional, Tuple
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
from src.config.server_settings import (  # noqa: E402
    ENV_BIND_HOST,
    bind_is_loopback,
    inherited_socket_bind,
    network_bind,
)

def log(msg: str) -> None:
    print(f"[SHEPHERD] {msg}")

def run(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check)

def serve(cmd: List[str]) -> int:
    """Run the server beside this process (the browser opener's) until it exits;
    its exit status.

    **A Ctrl-C waits for the server, never kills it.** The server is in this
    terminal's process group and gets the same SIGINT; its shutdown stops the
    GPU job it runs, which leads a process group of its own and so no longer
    gets the terminal's signal (`src/api/main.py`, review of 8242b47).
    `subprocess.run` would kill the server 0.25 s after the first Ctrl-C,
    before that shutdown ends, and leave the job running. A second Ctrl-C
    makes uvicorn quit at once, without its shutdown.
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

def _open_browser_when_ready(url: str, health_url: str, progress_every: float = 30.0) -> None:
    """Poll ``health_url`` until it answers, then open ``url`` once.

    Runs in a daemon thread beside the server. **There is no deadline.**
    uvicorn runs the app's lifespan, where an eager pipeline build loads,
    before it binds the socket, so the first answer is the end of startup
    however long the load takes. A fixed 60 s bound only guessed that time,
    and gave up on a start that loads a large SP table eagerly (about 84 s
    on a GB10; BACKLOG item 17). The wait cannot outlive the server: when it
    exits, ``run`` returns, ``main`` exits and this daemon thread ends with
    the process. A line about every ``progress_every`` seconds says the
    server is still starting.

    ``/health``, not ``/ready``: ``/ready`` stays 503 after a failed startup,
    and the dashboard is where an operator then reloads a pipeline.
    """
    started = time.monotonic()
    next_progress = progress_every
    while True:
        try:
            resp = urlopen(health_url, timeout=3)
            if resp.status == 200:
                log(f"Server ready — opening browser: {url}")
                webbrowser.open(url)
                return
        except (URLError, OSError):
            pass
        elapsed = time.monotonic() - started
        if elapsed >= next_progress:
            log(f"Server still starting ({elapsed:.0f} s); the browser opens when {health_url} answers")
            next_progress = elapsed + progress_every
        time.sleep(1.0)

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

def pip_install(spec: str, name: str, reinstall: bool = False, no_deps: bool = False) -> bool:
    args = [sys.executable, "-m", "pip", "install", "-U"]
    if reinstall:
        args.extend(["--force-reinstall", "-I"])
    if no_deps:
        args.append("--no-deps")
    args.append(spec)
    log(f"Installing {name}: {spec}")
    try:
        run(args, check=True)
        return True
    except subprocess.CalledProcessError:
        log(f"WARNING: pip install failed for {name}")
        return False

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

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shep_launch", description="SHEPHERD launcher for optional accelerators", formatter_class=argparse.RawTextHelpFormatter,
                                epilog="Arguments after -- go to the main app unchanged, e.g.  shep_launch.py --no-browser -- --port 8264")
    g = p.add_argument_group("Accelerator flags (opt-in like SD WebUI)")
    g.add_argument("--flash-attn", action="store_true", help="Enable FlashAttention (Windows/x86; skipped on ARM)")
    g.add_argument("--xformers", action="store_true", help="Enable xFormers memory-efficient attention")
    g.add_argument("--sage-attn", action="store_true", help="Enable SageAttention (community plugin)")
    g.add_argument("--cudnn-sdpa", action="store_true", help="Prefer cuDNN SDPA (default if nothing else selected)")
    g.add_argument("--torch-sdpa", action="store_true", help="Prefer vanilla Torch SDPA")
    g.add_argument("--naive-attn", action="store_true", help="Force naive/matmul attention as last resort")
    r = p.add_argument_group("Reinstall/behavior controls")
    r.add_argument("--reinstall-flash-attn", action="store_true")
    r.add_argument("--reinstall-xformers", action="store_true")
    r.add_argument("--reinstall-sage-attn", action="store_true")
    r.add_argument("--no-auto-install", action="store_true", help="Do not pip-install automatically; only check")
    o = p.add_argument_group("Ordering and plugins")
    o.add_argument("--attention-order", type=str, default="", help="Comma list, e.g. flash_attn,xformers,cudnn_sdpa,torch_sdpa,naive")
    o.add_argument("--plugin", type=str, default="", help="Custom plugin path 'module.sub:Class' (will be checked)")
    d = p.add_argument_group("Diagnostics")
    d.add_argument("--print-plan", action="store_true", help="Print chosen plan and exit code")
    d.add_argument("--dry-run", action="store_true", help="Simulate actions without installing or launching")
    d.add_argument("--skip-launch", action="store_true", help="Do not start main app after setup")
    m = p.add_argument_group("Main app")
    m.add_argument("--entry", type=str, default=DEFAULT_ENTRY, help="Python module to run with -m (default: uvicorn)")
    m.add_argument("--no-browser", action="store_true", help="Do not open a browser once the server is ready. On Linux/macOS the launcher then\nreplaces itself with the server (exec) instead of waiting beside it.")
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
    """Whether this process may become the server by ``exec``.

    POSIX only. There ``execve`` replaces the process image and keeps the PID,
    so a service manager's main PID becomes the server and its signals reach
    it. On Windows ``os.exec*`` starts a new process and ends this one: the PID
    changes and a supervisor loses track, so the launcher waits instead.
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

    # **The bind guard** (SPEC_4 §3.1), before anything is installed or
    # started: what uvicorn will listen on (`bind_plan`) other than loopback is
    # refused unless SHEPHERD_ACCEPT_UNAUTHENTICATED_NETWORK_BIND=1 accepts the
    # risk (no control: D8 still refuses operator controls off loopback). Only
    # for the uvicorn entry, whose options this launcher can read. The host is
    # passed to application startup, which checks it again; no host, none.
    bind: Optional[BindPlan] = None
    os.environ.pop(ENV_BIND_HOST, None)
    if args.entry == "uvicorn":
        bind = bind_plan(UVICORN_DEFAULT_ARGS + passthrough, os.environ)
        if bind.decision is not None and not bind.decision.allowed:
            log(f"ERROR: {bind.decision.message}")
            return 2
        if bind.host is not None:
            os.environ[ENV_BIND_HOST] = bind.host

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
    resolved: List[str] = []
    def try_enable(mod_key: str, import_name: str, reinstall_flag: bool, display: str, allow_on_arm: bool = True, no_deps: bool = False) -> bool:
        if is_arm() and not allow_on_arm:
            log(f"Skip {display} on ARM architecture"); return False
        if have_module(import_name):
            log(f"{display} already available"); return True
        if args.no_auto_install:
            log(f"{display} not installed; --no-auto-install set, skipping install"); return False
        spec = choose_spec(table, mod_key, torch_v=torch_v, cuda_v=cuda_v, py_v=py_v, os_name=os_name, arch=arch)
        if not spec:
            log(f"No install spec found for {display}; skipping"); return False
        if args.dry_run:
            log(f"[dry-run] Would install {display}: {spec}"); return False
        ok = pip_install(spec, display, reinstall=reinstall_flag, no_deps=no_deps)
        return ok and have_module(import_name)
    if "flash_attn" in order:
        enabled = try_enable("flash_attn", "flash_attn", args.reinstall_flash_attn, "FlashAttention", allow_on_arm=False, no_deps=True)
        if not enabled:
            order = [x for x in order if x != "flash_attn"]
            os.environ["FLASHATTN_FORCE_DISABLE"] = "1"
    if "xformers" in order:
        enabled = try_enable("xformers", "xformers", args.reinstall_xformers, "xFormers")
        if not enabled:
            order = [x for x in order if x != "xformers"]
    if "sage_attn" in order:
        enabled = try_enable("sage_attn", "sage_attention", args.reinstall_sage_attn, "SageAttention")
        if not enabled:
            order = [x for x in order if x != "sage_attn"]
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

    # Computed before the access points are logged: those lines used to say port
    # 8000 whatever --port was passed, so the unit's journal told an operator to
    # bookmark a port nothing listened on.
    base_url = display_base_url(UVICORN_DEFAULT_ARGS + passthrough)

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
    if args.print_plan or args.dry_run or args.skip_launch:
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
        cmd = ([sys.executable, "-m", "uvicorn", UVICORN_APP] + UVICORN_DEFAULT_ARGS
               + ["--no-access-log"] + passthrough + UVICORN_FORCED_ARGS)
    else:
        cmd = [sys.executable, "-m", args.entry] + passthrough
    # **Headless: become the server.** The only reason to keep this process
    # alive is the browser thread below, and --no-browser has none. Everything
    # above -- allocator, attention, plugin, retrieval -- is already in
    # os.environ, so exec hands it over whole. No launcher is left resident,
    # the service's main PID is the server, and Restart Backend re-execs it in
    # place as it would a bare uvicorn start.
    if args.no_browser and can_hand_over():
        return exec_server(cmd)
    # Auto-open browser once server is ready (polls /health until it answers)
    if not args.no_browser and not has_urls:
        log("No browser is opened: the server has no URL this launcher knows")
    elif not args.no_browser:
        opener = threading.Thread(
            target=_open_browser_when_ready,
            args=(f"{base_url}/ui", f"{base_url}/health"),
            daemon=True,
        )
        opener.start()

    return serve(cmd)

if __name__ == "__main__":
    sys.exit(main())