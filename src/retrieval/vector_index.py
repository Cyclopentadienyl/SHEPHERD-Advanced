import platform
import importlib

def resolve_backend():
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        # Prefer FAISS GPU on x86
        if importlib.util.find_spec("faiss") or importlib.util.find_spec("faiss_gpu"):
            return "faiss_gpu"
    # Default (and ARM path) → HNSWLIB
    return "hnswlib"
