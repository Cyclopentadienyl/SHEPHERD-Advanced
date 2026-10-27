#!/usr/bin/env bash
set -euo pipefail
umask 077
E=/home/user/batch3-acceptance-be0ee1a-01
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for name in acceptance-check.txt check-exit.txt; do
  if [[ -e "$E/$name" || -L "$E/$name" ]]; then
    echo "已存在，保留原檔並停止：$E/$name"; exit 1
  fi
done
set +e
/home/user/Desktop/SHEPHERD-Advanced/.venv/bin/python -I "$HERE/03-check.py" \
  2>&1 | tee "$E/acceptance-check.txt"
codes=("${PIPESTATUS[@]}")
set -e
printf '%s %s\n' "${codes[0]}" "${codes[1]}" > "$E/check-exit.txt"
if (( codes[0] != 0 || codes[1] != 0 )); then
  echo '核對未通過，保留輸出交給 Codex；不要重跑覆寫。'; exit 1
fi
