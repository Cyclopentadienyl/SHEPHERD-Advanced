import json, os, platform, sys, importlib
from pathlib import Path

REPORT = {"errors": [], "warnings": [], "info": []}

def add(level, msg):
    REPORT[level].append(msg)

def check_torch(min_ver="2.9.0"):
    try:
        import torch
        from packaging import version
        add("info", f"Torch version: {torch.__version__}")
        if version.parse(torch.__version__) < version.parse(min_ver):
            add("errors", f"Torch >= {min_ver} required, found {torch.__version__}")
        if torch.cuda.is_available():
            add("info", f"CUDA available: {torch.version.cuda}")
            try:
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    add("info", "SDPA (Scaled Dot Product Attention): Available")
                else:
                    add("warnings", "SDPA: Not available in this Torch version")
            except Exception as e:
                add("warnings", f"SDPA probe failed: {e}")
        else:
            add("warnings", "CUDA not available (running on CPU)")
    except Exception as e:
        add("errors", f"import torch failed: {e}")

def check_pkg(name):
    try:
        importlib.import_module(name)
        add("info", f"{name} OK")
        return True
    except Exception as e:
        add("warnings", f"{name} not available: {e}")
        return False

def check_flash_attn():
    if os.environ.get("FLASHATTN_FORCE_DISABLE"):
        add("info", "FLASHATTN_FORCE_DISABLE set; skipping flash-attn check")
        return
    try:
        flash_attn = importlib.import_module("flash_attn")
        add("info", f"flash-attn OK (v{flash_attn.__version__})")
    except Exception as e:
        add("warnings", f"flash-attn not available: {e}")

def main():
    add("info", f"Python: {sys.version.split()[0]}")
    add("info", f"OS: {platform.system().lower()}  Arch: {platform.machine()}")

    check_torch()
    if not check_pkg("xformers"):
        add("warnings", "xformers not installed (optional but recommended)")
    faiss_available = (
        check_pkg("faiss") or 
        check_pkg("faiss_gpu") or 
        check_pkg("faiss_cpu")
    )
    if not faiss_available:
        add("warnings", "No FAISS backend found (faiss/faiss-gpu/faiss-cpu)")
    check_pkg("hnswlib")
    check_flash_attn()

    print(json.dumps(REPORT, indent=2))

if __name__ == "__main__":
    main()