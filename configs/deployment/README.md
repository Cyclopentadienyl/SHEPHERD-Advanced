# Deployment Configuration

This folder previously contained platform-specific YAML files. As of v3.1, all deployment
configuration has been **consolidated into a single file**:

```
configs/deployment.yaml   <-- Unified configuration (all platforms)
```

## Configuration Structure

The unified `deployment.yaml` uses a **defaults + platform overrides** pattern:

```yaml
defaults:
  # Shared settings for all platforms
  torch:
    version: "2.9.0"
  attention_backend:
    prefer: [torch_sdpa, naive]

platforms:
  linux_x86_64:
    # Overrides for Linux x86
    attention_backend:
      prefer: [flash_attn, torch_sdpa, naive]

  linux_aarch64:
    # Overrides for ARM (DGX Spark)
    attention_backend:
      prefer: [cudnn_sdpa, torch_sdpa, naive]

  windows_x86_64:
    # Overrides for Windows
    ...
```

## Platform Detection

The launcher (`scripts/launch/shep_launch.py`) automatically detects the platform
using `os + arch` (e.g., `linux_x86_64`, `linux_aarch64`, `windows_x86_64`) and
applies the appropriate configuration overrides.

## Environment Variable Overrides

Any setting can be overridden via environment variables:

```bash
# Override attention backend order
export ATTENTION_ORDER=torch_sdpa,naive

# Override retrieval backend
export SHEPHERD_RETRIEVAL_BACKEND=hnswlib
```

## Accelerator Installation

Optional accelerators (flash-attn, xformers, sage-attn) are configured in:

```
configs/accelerators.json   <-- Installation specs per platform/Python/CUDA version
```

Install via launch flags:

```bash
./launch_shepherd.sh --flash-attn    # Install and enable FlashAttention-2
./launch_shepherd.sh --xformers      # Install and enable xFormers
```

## Migration from v3.0

If you have custom modifications to the old platform-specific files, merge them
into the appropriate `platforms:` section in `deployment.yaml`.

Old files (removed):
- `linux_x86.yaml` → `platforms.linux_x86_64`
- `linux_arm_dgx.yaml` → `platforms.linux_aarch64`
- `windows.yaml` → `platforms.windows_x86_64`
