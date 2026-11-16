#!/usr/bin/env bash
###############################################################################
# SHEPHERD-Advanced Launcher Entry Point (Linux/ARM)
###############################################################################
#
# This is a convenience wrapper that:
# 1. Finds the repository it belongs to, wherever it is started from (a
#    desktop entry, another directory, a symlink to it), and names everything
#    in it by absolute path, so the virtual environment beside it is found
#    (UI-06a). It does not change directory itself: the launcher reads a
#    relative --settings-file against the directory it was started in, then
#    changes to the repository, where uvicorn imports `src`
# 2. Activates the virtual environment
# 3. Passes all arguments to scripts/launch/shep_launch.py, after
#    --local-entry (the local start: on a loopback bind the operator controls
#    are on for this computer unless SHEPHERD_DEPLOYMENT_MODE says otherwise;
#    without a terminal the output goes to a log file), replacing itself with
#    it (exec), so the PID this script was started as is the launcher's,
#    which then becomes the server's (it execs into uvicorn): a SIGTERM sent
#    to it alone stops the server, which stops its GPU job
# 4. Allows users to set COMMANDLINE_ARGS environment variable
#
# Usage Examples:
#   ./launch_shepherd.sh
#   ./launch_shepherd.sh --xformers
#   ./launch_shepherd.sh --print-plan
#   ./launch_shepherd.sh --dry-run
#   ./launch_shepherd.sh --install-desktop-entry
#   ./launch_shepherd.sh --settings-file /abs/slot-b/deployment.json
#
#   export COMMANDLINE_ARGS="--xformers --cudnn-sdpa"
#   ./launch_shepherd.sh
###############################################################################

set -euo pipefail

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

log() {
    echo -e "${CYAN}[SHEPHERD]${NC} $*"
}

success() {
    echo -e "${GREEN}  ✓${NC} $*"
}

warning() {
    echo -e "${YELLOW}WARNING:${NC} $*"
}

# Whether any standard stream is a terminal (the launcher's own rule,
# shep_launch.has_terminal): a start from the desktop entry has none.
has_terminal() {
    [[ -t 0 || -t 1 || -t 2 ]]
}

# The state directory the launcher logs to (src/utils/launch_instance.py's
# state_directory); empty when nothing names one.
state_dir() {
    if [[ "${SHEPHERD_STATE_DIR:-}" == /* ]]; then
        printf '%s' "$SHEPHERD_STATE_DIR"
    elif [[ "${XDG_STATE_HOME:-}" == /* ]]; then
        printf '%s' "$XDG_STATE_HOME/shepherd-advanced"
    elif [[ "${HOME:-}" == /* ]]; then
        if [[ "$(uname -s)" == Darwin ]]; then
            printf '%s' "$HOME/Library/Application Support/shepherd-advanced"
        else
            printf '%s' "$HOME/.local/state/shepherd-advanced"
        fi
    fi
}

# Each replacement quoted: since bash 5.2 (patsub_replacement) an unquoted &
# in one stands for the text matched.
html_escape() {
    local text=${1//&/"&amp;"}
    text=${text//</"&lt;"}
    text=${text//>/"&gt;"}
    text=${text//\"/"&quot;"}
    printf '%s' "$text"
}

# Without a terminal nobody reads the message: append it to the launch log and
# open a local page that says what failed, what to do and where the log is.
error_page() {
    local what=$1 remedy=$2 dir log page
    dir=$(state_dir)
    [[ -n "$dir" ]] || return 0
    log="$dir/logs/launch.log"
    page="$dir/logs/launch-error.html"
    mkdir -p -m 700 "$dir/logs" || return 0
    printf '[SHEPHERD] ERROR: %s %s\n' "$what" "$remedy" >>"$log" 2>/dev/null || true
    {
        printf '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
        printf '<title>SHEPHERD-Advanced did not start</title>\n'
        printf '<style>body { font: 15px/1.5 system-ui, sans-serif; color: #1f2430; '
        printf 'background: #fff; max-width: 720px; margin: 40px auto; padding: 0 16px; }\n'
        printf 'h1 { font-size: 20px; font-weight: 600; } dt { font-size: 14px; '
        printf 'font-weight: 600; margin-top: 16px; } dd { margin: 4px 0 0; }\n'
        printf 'code { font-size: 14px; word-break: break-all; }</style></head>\n'
        printf '<body><h1>SHEPHERD-Advanced did not start</h1>\n'
        printf '<dl><dt>What failed</dt><dd>%s</dd>\n' "$(html_escape "$what")"
        printf '<dt>What to do</dt><dd>%s</dd>\n' "$(html_escape "$remedy")"
        printf '<dt>The full log</dt><dd><code>%s</code></dd></dl>\n' "$(html_escape "$log")"
        printf '</body></html>\n'
    } >"$page" 2>/dev/null || return 0
    chmod 600 "$page" 2>/dev/null || true
    if command -v xdg-open >/dev/null 2>&1; then
        (xdg-open "file://$page" >/dev/null 2>&1 &) || true
    elif command -v open >/dev/null 2>&1; then
        (open "$page" >/dev/null 2>&1 &) || true
    fi
}

fail() {
    echo -e "${RED}ERROR:${NC} $1"
    [[ -n "${2:-}" ]] && warning "$2"
    if ! has_terminal; then
        error_page "$1" "${2:-}"
    fi
    exit 1
}

###############################################################################
# Main
###############################################################################
echo -e "${CYAN}SHEPHERD-Advanced Launcher${NC}"
echo ""

# This script's own directory, symlinks followed one by one (a portable
# `readlink -f`: macOS's readlink has no -f): the repository, named by
# absolute path below. The working directory is left as it was started; the
# launcher changes to the repository once it has read its arguments.
script_path="${BASH_SOURCE[0]}"
while [[ -L "$script_path" ]]; do
    link_dir="$(cd -P "$(dirname "$script_path")" && pwd)"
    script_path="$(readlink "$script_path")"
    [[ "$script_path" == /* ]] || script_path="$link_dir/$script_path"
done
REPO_ROOT="$(cd -P "$(dirname "$script_path")" && pwd)"

# Check if virtual environment exists
if [[ ! -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    fail "The virtual environment $REPO_ROOT/.venv was not found." \
         "Run deploy.sh in $REPO_ROOT first, then start SHEPHERD-Advanced again."
fi

# Activate virtual environment
log "Activating virtual environment..."
source "$REPO_ROOT/.venv/bin/activate"
success "Virtual environment activated"
echo ""

# Check if shep_launch.py exists
if [[ ! -f "$REPO_ROOT/scripts/launch/shep_launch.py" ]]; then
    fail "$REPO_ROOT/scripts/launch/shep_launch.py was not found." \
         "This checkout is incomplete: get it again, then run deploy.sh."
fi

# Launch the Python launcher with all arguments, in this process's place
# (round 3 of the review fixes, P3-2): a launcher started as this script's
# child was not reached by a SIGTERM sent to the PID that was started.
log "Launching SHEPHERD via shep_launch.py..."
echo ""
exec python "$REPO_ROOT/scripts/launch/shep_launch.py" --local-entry "$@"
