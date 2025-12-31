# Windows (CMD) scripts

为了最大化兼容性与稳定性，这里提供纯 CMD 版部署与初始化脚本：

- `scripts\deploy\windows_x86.cmd`
  ```cmd
  rem 使用默认配置
  scripts\deploy\windows_x86.cmd

  rem 或指定配置 yaml
  scripts\deploy\windows_x86.cmd configs\deployment\windows.yaml
  ```

  支持可选环境变量：
  - `PYTHON`：指定 Python 可执行（默认 `python`）。
  - `FLASHATTN_WHEEL`：如需从本地 wheel 安装 flash-attn，设为 `.whl` 路径。

- `scripts\setup_env.cmd`
  ```cmd
  rem 使用默认 requirements_windows.txt
  scripts\setup_env.cmd

  rem 或指定其它 requirements 文件
  scripts\setup_env.cmd requirements_arm.txt
  ```
