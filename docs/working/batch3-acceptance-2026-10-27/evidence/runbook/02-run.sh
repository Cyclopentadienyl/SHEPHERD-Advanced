#!/usr/bin/env bash
set -euo pipefail
umask 077
REPO=/home/user/Desktop/SHEPHERD-Advanced
E=/home/user/batch3-acceptance-be0ee1a-01
WS=/home/user/shepherd-data/workspaces/m5_webui_demo_01
CKPT="$WS/runs/gat/20261024-173909_ui-live-stop-02_c1d9cf/checkpoints/last.pt"
cd "$REPO"
test -f "$E/tested-checkout.txt"
test "$(git rev-parse HEAD)" = be0ee1a662db8d16a85533c0394244ffcb64ae29
git diff --quiet
git diff --cached --quiet
for mode in C D; do
  for p in "$E/$mode.log" "$E/$mode-exit.txt" "$E/$mode-outcome.json"; do
    if [[ -e "$p" || -L "$p" ]]; then
      echo "已存在，保留原檔並停止：$p"; exit 1
    fi
  done
done
if pgrep -af '[u]vicorn|[s]hep_launch.py|[p]robe_deployment.py|[t]rain_model.py|[m]easure_scorer.py'; then
  echo '仍有相關程序，先停止服務／工作。'; exit 1
fi
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
test -z "$gpu_pids" || { printf '仍有 GPU 程序：\n%s\n' "$gpu_pids"; exit 1; }
sha256sum --quiet -c "$E/inputs.sha256"

export SHEPHERD_WORKSPACES_ROOT=/home/user/shepherd-data/workspaces
export SHEPHERD_REGISTRY=/home/user/shepherd-data/registry
export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
printf '%s\n' 'PYTORCH_ALLOC_CONF=backend:cudaMallocAsync' \
  '來源：本次 CLI 明確設定；不是 CLI 自動讀取 saved preset。' > "$E/allocator-setting.txt"

# 只呼叫正式量測入口；C 成功後才跑 D，不訓練、不啟動服務。
for mode in C D; do
  set +e
  "$REPO/.venv/bin/python" "$REPO/scripts/measure_scorer.py" \
    --mode "$mode" --checkpoint "$CKPT" --data-dir "$WS" \
    --cohort m5-closeout-demo-01 --registry "$SHEPHERD_REGISTRY" \
    --device cuda --batch-size 32 --record \
    --expect-checkpoint-sha256 568b109848430c92de8d13bd5e24d7836bc16fe32f17a088952be4b322a856bf \
    --expect-version-sha256 1c2c6050951155df6ccad39e07579c0df83536a35779f7ce916bda9561bb96c3 \
    --outcome "$E/$mode-outcome.json" 2>&1 | tee "$E/$mode.log"
  codes=("${PIPESTATUS[@]}")
  set -e
  printf '%s %s\n' "${codes[0]}" "${codes[1]}" > "$E/$mode-exit.txt"
  printf '\nMode %s：量測 exit=%s；tee exit=%s\n' "$mode" "${codes[0]}" "${codes[1]}"
  if (( codes[0] != 0 || codes[1] != 0 )); then
    echo '本輪停止，保留所有輸出。不要重跑覆寫；請把 log 交給 Codex。'; exit 1
  fi
done
echo 'C、D 均已結束。接著核對報告與帳本，尚未還原 checkout。'
