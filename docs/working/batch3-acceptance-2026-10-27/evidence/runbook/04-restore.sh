#!/usr/bin/env bash
set -euo pipefail
umask 077
REPO=/home/user/Desktop/SHEPHERD-Advanced
E=/home/user/batch3-acceptance-be0ee1a-01
cd "$REPO"
test -f "$E/checkout-before.txt"
for name in checkout-restored.txt input-recheck.txt; do
  if [[ -e "$E/$name" || -L "$E/$name" ]]; then
    echo "已存在，保留原檔並停止：$E/$name"; exit 1
  fi
done
if pgrep -af '[u]vicorn|[s]hep_launch.py|[p]robe_deployment.py|[t]rain_model.py|[m]easure_scorer.py'; then
  echo '仍有相關程序，先不要切換 checkout。'; exit 1
fi
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$gpu_pids" || { printf '仍有 GPU 程序：\n%s\n' "$gpu_pids"; exit 1; }
git diff --quiet
git diff --cached --quiet
test "$(git rev-parse HEAD)" = be0ee1a662db8d16a85533c0394244ffcb64ae29
branch="$(sed -n '1p' "$E/checkout-before.txt")"
sha="$(sed -n '2p' "$E/checkout-before.txt")"
if [[ "$branch" == '(detached)' ]]; then
  git switch --detach "$sha"
else
  test "$(git rev-parse "refs/heads/$branch")" = "$sha"
  git switch "$branch"
fi
{
  git symbolic-ref -q --short HEAD || echo '(detached)'
  git rev-parse HEAD
} > "$E/checkout-restored.txt"
diff -- "$E/checkout-before.txt" "$E/checkout-restored.txt"
sha256sum -c "$E/inputs.sha256" | tee "$E/input-recheck.txt"
cat "$E/checkout-restored.txt"
echo 'checkout 與既有輸入已核對；新增 C/D 報告、帳本紀錄及所有證據均保留。'
