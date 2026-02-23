# scripts/launch/shep_launch.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys, textwrap, threading, time, webbrowser
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import shlex

REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/launch -> scripts -> REPO_ROOT
CONFIG_DIR = REPO_ROOT / "configs"  # aligned with v3 structure
ACCEL_TABLE = CONFIG_DIR / "accelerators.json"
DEPLOYMENT_CONFIG = CONFIG_DIR / "deployment.yaml"
DEFAULT_ENTRY = "uvicorn"
UVICORN_APP = "src.api.main:app"
UVICORN_DEFAULT_ARGS = ["--host", "0.0.0.0", "--port", "8000"]

def log(msg: str) -> None:
    print(f"[SHEPHERD] {msg}")

def run(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check)

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
    p = argparse.ArgumentParser(prog="shep_launch", description="SHEPHERD launcher for optional accelerators", formatter_class=argparse.RawTextHelpFormatter)
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
    m.add_argument("--", dest="passthrough", nargs=argparse.REMAINDER, help="Arguments passed to the main app after --")
    return p
    
def collect_args_from_env_and_cli() -> List[str]:
    env = os.getenv("SHEP_COMMANDLINE_ARGS") or os.getenv("COMMANDLINE_ARGS") or ""
    env_args = shlex.split(env)
    cli_args = sys.argv[1:]
    return env_args + cli_args

def main() -> int:
    parser = build_parser()
    merged_args = collect_args_from_env_and_cli()
    args = parser.parse_args(merged_args)
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

    # Collect passthrough args
    passthrough: List[str] = []
    if args.passthrough:
        passthrough = list(args.passthrough)
        if passthrough and passthrough[0] == "--":
            passthrough = passthrough[1:]

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
    if args.print_plan or args.dry_run or args.skip_launch:
        return 0

    # Determine host/port for URL display
    host = "127.0.0.1"
    port = "8000"
    uvi_args = UVICORN_DEFAULT_ARGS + passthrough
    for i, a in enumerate(uvi_args):
        if a == "--port" and i + 1 < len(uvi_args):
            port = uvi_args[i + 1]
        elif a == "--host" and i + 1 < len(uvi_args):
            h = uvi_args[i + 1]
            if h != "0.0.0.0":
                host = h

    base_url = f"http://{host}:{port}"

    # Print web interface endpoints
    print(textwrap.dedent(f"""\
    ===  Web Interfaces  ===
      Swagger UI (API docs) : {base_url}/docs
      Gradio Dashboard      : {base_url}/ui
    ========================
    """))

    # Auto-open Gradio UI in default browser after a short delay
    def _open_browser() -> None:
        time.sleep(3)  # wait for uvicorn to start
        try:
            webbrowser.open(f"{base_url}/ui")
        except Exception:
            pass  # non-critical; ignore if no browser available

    threading.Thread(target=_open_browser, daemon=True).start()

    # Build launch command
    if args.entry == "uvicorn":
        cmd = [sys.executable, "-m", "uvicorn", UVICORN_APP] + UVICORN_DEFAULT_ARGS + passthrough
    else:
        cmd = [sys.executable, "-m", args.entry] + passthrough
    res = run(cmd)
    return res.returncode

if __name__ == "__main__":
    sys.exit(main())