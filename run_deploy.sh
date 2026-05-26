#!/usr/bin/env bash
# Double-click friendly launcher: runs deploy.sh from the project root, mirrors
# all output to deploy.log, and pauses at the end so a file-manager-spawned
# terminal window stays open long enough to read the result.
cd "$(dirname "$0")" || { echo "[ERROR] cannot cd to script directory"; read -rn1 -p "Press any key to close..."; exit 1; }

./deploy.sh 2>&1 | tee deploy.log
status=${PIPESTATUS[0]}  # exit code of deploy.sh, not tee

echo
if [ "$status" -eq 0 ]; then
    echo "[OK] deploy.sh finished (exit 0). Full output saved to deploy.log"
else
    echo "[FAIL] deploy.sh exited with code $status. See deploy.log for details."
fi
read -rn1 -p "Press any key to close..."
exit "$status"
