#!/usr/bin/env bash
set -euo pipefail
umask 077
REPO=/home/user/Desktop/SHEPHERD-Advanced
E=/home/user/batch3-acceptance-be0ee1a-01
SHA=be0ee1a662db8d16a85533c0394244ffcb64ae29
cd "$REPO"
if [[ -e "$E" || -L "$E" ]]; then
  echo "已存在，保留原檔並停止：$E"; exit 1
fi
git diff --quiet
git diff --cached --quiet
if pgrep -af '[u]vicorn|[s]hep_launch.py|[p]robe_deployment.py|[t]rain_model.py|[m]easure_scorer.py'; then
  echo '仍有相關程序，先停止服務／工作。'; exit 1
fi
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$gpu_pids" ]]; then
  printf '仍有 GPU 程序：\n%s\n' "$gpu_pids"; exit 1
fi
mkdir "$E"
{
  git symbolic-ref -q --short HEAD || echo '(detached)'
  git rev-parse HEAD
} > "$E/checkout-before.txt"
git status --porcelain=v1 > "$E/git-status-before.txt"

# 留下既有資料摘要、舊帳本及使用紀錄；不複製或改寫模型。
"$REPO/.venv/bin/python" -I - "$E" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
e = Path(sys.argv[1])
w = Path('/home/user/shepherd-data/workspaces/m5_webui_demo_01')
v = Path('/home/user/shepherd-data/registry/versions/m5-closeout-demo-01')
c = w / 'runs/gat/20261024-173909_ui-live-stop-02_c1d9cf/checkpoints/last.pt'
ledger = c.parent / 'evaluations.json'
usage = v / 'reports/usage.jsonl'
def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
expected = {
    c: '568b109848430c92de8d13bd5e24d7836bc16fe32f17a088952be4b322a856bf',
    v / 'cohort.json': '1c2c6050951155df6ccad39e07579c0df83536a35779f7ce916bda9561bb96c3',
    v / 'samples.json': '91b77e24e5fc7023894b5791df156f3c9c5f552e6e2e82168117fae77c45ca6e',
}
for p, sha in expected.items():
    if digest(p) != sha:
        raise SystemExit(f'輸入與預先核對的版本不同，停止：{p}')
files = sorted({p for root in (w, v) for p in root.rglob('*')
                if p.is_file() and p not in (ledger, usage)})
with (e / 'inputs.sha256').open('x') as f:
    for p in files:
        f.write(f'{digest(p)}  {p}\n')
shutil.copyfile(ledger, e / 'ledger-before.json')
shutil.copyfile(usage, e / 'usage-before.jsonl')
print(f'已保存 {len(files)} 份既有檔案的摘要；舊帳本有 {len(json.loads(ledger.read_text())["records"])} 筆。')
PY

git fetch origin claude/blueprint-after-m5
git cat-file -e "$SHA^{commit}"
git switch --detach "$SHA"
test "$(git rev-parse HEAD)" = "$SHA"
git diff --quiet
git diff --cached --quiet
git rev-parse HEAD > "$E/tested-checkout.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv > "$E/gpu-before.txt"
sha256sum "$REPO/pyproject.toml" "$REPO/uv.lock" > "$E/dependencies.sha256"
cp -R -- "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" "$E/runbook"
printf '\n準備完成。原 checkout：\n'
cat "$E/checkout-before.txt"
printf '\n受測版本：%s\n證據：%s\n尚未開始量測；不需安裝套件。\n' "$SHA" "$E"
