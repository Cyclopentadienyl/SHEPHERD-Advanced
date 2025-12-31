# Root-level Windows CMD entry points

为便于发现与使用，Windows 入口脚本放在**仓库根目录**：

- `deploy.cmd`  
  - 自动创建/复用 `.venv/`（根目录），安装 `requirements_windows.txt`，可选安装本地 `FLASHATTN_WHEEL`，运行验证，并尝试构建索引：  
    ```cmd
    deploy.cmd
    deploy.cmd configs\deployment\windows.yaml
    set FLASHATTN_WHEEL=C:\path\to\flash_attn-*.whl
    deploy.cmd
    ```

- `setup_env.cmd`  
  - 仅做环境初始化：创建 `.venv` 并安装给定的 requirements：  
    ```cmd
    setup_env.cmd
    setup_env.cmd requirements_arm.txt
    ```

> 两个脚本都使用 `.\.venv\Scripts\python.exe` 直接执行，无需 `activate`。

## .gitignore 建议
将 `.venv/` 排除出 Git：见随附的 `.gitignore_additions.txt`，或在你的 `.gitignore` 中加入：
```
.venv/
```

## Linux 部署保持不变
- `scripts/deploy/linux_x86.sh`
- `scripts/deploy/dgx_spark_arm.sh`

Windows 侧统一用根目录的 `deploy.cmd / setup_env.cmd`。
