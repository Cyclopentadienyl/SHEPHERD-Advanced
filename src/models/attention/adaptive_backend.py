import os
import platform
from importlib import import_module

import torch
import torch.nn.functional as F

# Attn order from env (comma separated). Default prefers plugin > FA2 > xFormers > cuDNN > SDPA > math > naive
_DEFAULT_ORDER = [
    "plugin", "flash_attn", "xformers", "cudnn_sdpa", "torch_sdpa", "math", "naive"
]

def _is_arm() -> bool:
    arch = platform.machine().lower()
    return arch in ("aarch64", "arm64", "armv8l")

def _has_flash_attn() -> bool:
    if _is_arm():
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False

def _has_xformers() -> bool:
    try:
        import xformers.ops as _  # noqa: F401
        return True
    except Exception:
        return False

def _has_cudnn_sdpa() -> bool:
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend  # torch>=2.1
        # If any CUDA is present, consider CUDNN available; real selection happens in context manager
        return torch.cuda.is_available()
    except Exception:
        return False

def _has_torch_sdpa() -> bool:
    try:
        # Always exists on torch>=2.0 CPU/GPU; we still check torch import
        return True
    except Exception:
        return False

def _load_plugin(path: str):
    """
    ATTENTION_PLUGIN="module.sub:ClassName"
    Plugin must implement .supports() -> bool and .forward(q,k,v,attn_mask=None,is_causal=False)
    """
    if not path or ":" not in path:
        return None
    mod_name, cls_name = path.split(":", 1)
    mod = import_module(mod_name)
    cls = getattr(mod, cls_name)
    impl = cls()
    return impl if getattr(impl, "supports", lambda: False)() else None

class AdaptiveAttentionBackend:
    """
    Pluggable attention backend with runtime detection and OS/arch awareness.
    """
    def __init__(self):
        order_env = os.environ.get("ATTENTION_ORDER")
        self.order = [s.strip() for s in order_env.split(",")] if order_env else list(_DEFAULT_ORDER)

        # Optional plugin hook
        self.plugin = _load_plugin(os.environ.get("ATTENTION_PLUGIN", ""))

        # Honor forced disable for FA2
        self.force_disable_fa = bool(os.environ.get("FLASHATTN_FORCE_DISABLE"))

        self.impl_name = None
        self._resolve()

    def _resolve(self):
        for name in self.order:
            if name == "plugin" and self.plugin:
                self.impl_name = "plugin"
                break
            if name == "flash_attn" and not self.force_disable_fa and _has_flash_attn():
                self.impl_name = "flash_attn"
                break
            if name == "xformers" and _has_xformers():
                self.impl_name = "xformers"
                break
            if name == "cudnn_sdpa" and _has_cudnn_sdpa():
                self.impl_name = "cudnn_sdpa"
                break
            if name == "torch_sdpa" and _has_torch_sdpa():
                self.impl_name = "torch_sdpa"
                break
            if name == "math":
                self.impl_name = "math"
                break
            if name == "naive":
                self.impl_name = "naive"
                break
        if self.impl_name is None:
            self.impl_name = "naive"
        print(f"[AttentionBackend] selected: {self.impl_name}")

    def name(self):
        return self.impl_name

    # Unified call
    def __call__(self, q, k, v, attn_mask=None, is_causal=False):
        if self.impl_name == "plugin":
            return self.plugin.forward(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

        if self.impl_name == "flash_attn":
            from flash_attn import flash_attn_func
            return flash_attn_func(q, k, v, causal=is_causal)

        if self.impl_name == "xformers":
            import xformers.ops as xops
            return xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask, causal=is_causal)

        if self.impl_name == "cudnn_sdpa":
            # Prefer cuDNN, then fallthrough to other fused kernels
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel([SDPBackend.CUDNN, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

        if self.impl_name == "torch_sdpa":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

        # Fallbacks
        # "math" and "naive" both use PyTorch's reference path here
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
